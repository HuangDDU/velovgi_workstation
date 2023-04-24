import numpy as np
import os
import pickle
from scipy.sparse import csr_matrix
import seaborn as sns
import networkx as nx

from sklearn.metrics import pairwise_distances
from scanpy.neighbors import _get_indices_distances_from_dense_matrix, _get_sparse_matrix_from_indices_distances_numpy, _compute_connectivities_umap
from scvelo.preprocessing.neighbors import _set_neighbors_data
from scvelo.preprocessing.neighbors import _get_rep, _set_pca, get_duplicate_cells
from scvelo import logging as logg

import scanpy as sc
import scvelo as scv
import torch

from .metric.eval_utils import inner_cluster_coh

########################################################################################################################
#######################################################抽样与恢复相关######################################################
# 随机采样
def random_sample(adata, n):
    cell_num = adata.shape[0]
    index_list = np.random.choice(np.arange(cell_num), size=n, replace=False)

    return index_list


# 中心性采样
def graph_centrality_sample(adata, n, mode="subgraph"):
    G = nx.from_scipy_sparse_array(adata.obsp["connectivities"])
    if mode == "degree":
        centrality_dict = nx.degree_centrality(G)
    elif mode == "eigenvector":
        centrality_dict = nx.eigenvector_centrality(G)
    elif mode == "betweenness":
        centrality_dict = nx.betweenness_centrality(G)
    elif mode == "closeness":
        centrality_dict = nx.closeness_centrality(G)
    elif mode == "pagerank":
        centrality_dict = nx.pagerank(G)
    elif mode == "current_flow_betweenness":
        centrality_dict = nx.current_flow_betweenness_centrality(G)
    elif mode == "communicability_betweenness":
        centrality_dict = nx.communicability_betweenness_centrality(G)
    elif mode == "load":
        centrality_dict = nx.load_centrality(G)
    elif mode == "subgraph":
        centrality_dict = nx.subgraph_centrality(G)
    elif mode == "harmonic":
        centrality_dict = nx.harmonic_centrality(G)
    elif mode == "second_order":
        centrality_dict = nx.second_order_centrality(G)
    else:
        raise ValueError("mode=%s is invalid" % mode)

    centrality_pair_list_sorted = sorted(list(centrality_dict.items()), key=lambda x: x[1], reverse=True)
    index_list = list(map(lambda x: x[0], centrality_pair_list_sorted))[:n]

    return index_list


# 抽样
def sample(adata, fraction=0.5, mode="random"):
    n = int(adata.shape[0] * fraction)
    # 按照模式选择采样的细胞索引
    if mode == "random":
        index_list = random_sample(adata, n)
    else:
        index_list = graph_centrality_sample(adata, n, mode)
    # 提取细胞
    index_list = sorted(index_list)  # 这里排序防止后续出错
    subsample_adata = adata[index_list].copy()
    is_sampled_key = "is_sampled"
    adata.obs[is_sampled_key] = False
    adata.obs.loc[subsample_adata.obs.index, is_sampled_key] = True  # 只有这样才能修改
    adata.uns["sample_recover"] = {}
    adata.uns["sample_recover"]["index_list"] = index_list
    return subsample_adata, index_list


def get_all_index_list(adata, index_list, threshold=0.98, mode="threshold", max_iter=10):
    all_index_list = [index_list]

    # 邻接矩阵
    w = (adata.obsp["connectivities"] > 0).astype(int)

    # 需要恢复的元素
    n = adata.shape[0]
    v = np.zeros(n)
    v[index_list] = 1
    v = csr_matrix(v, dtype=int)
    v = v.T

    # 邻接矩阵调整一下， 对角线元素置为1，表示即使邻居没有被计算过，只要自己计算过了，后续都算计算过。
    w_adjust = w.copy()
    w_adjust[np.arange(n), np.arange(n)] = 1

    # 根据阈值确定平滑次数
    tmp_v = csr_matrix.dot(w_adjust, v)
    # print(tmp_v.A.T)

    iter = 1
    if mode == "all":
        # 尽量恢复所有细胞
        pre_bool_array = np.zeros(tmp_v.shape).astype("bool").flatten()
        now_bool_array = (tmp_v.A > 0).flatten()
        while not (pre_bool_array == now_bool_array).all():
            # 上一次平滑没有新细胞获得值时就停止
            tmp_index_list = list(np.flatnonzero(tmp_v.A))  # 上一次恢复的细胞序号保存下来
            all_index_list.append(tmp_index_list)
            tmp_v = csr_matrix.dot(w_adjust, tmp_v)
            pre_bool_array = now_bool_array
            now_bool_array = (tmp_v.A > 0).flatten()
            iter += 1
            if iter == max_iter:
                print("up to max iter", max_iter)
                break
        all_index_list = all_index_list[:-1]  # 丢弃最后一个重复计算的矩阵
    else:
        # 恢复到指定阈值就停止
        while tmp_v.data.shape[0] / n < threshold:
            tmp_index_list = list(np.flatnonzero(tmp_v.A))  # 上一次恢复的细胞序号保存下来
            all_index_list.append(tmp_index_list)
            tmp_v = csr_matrix.dot(w_adjust, tmp_v)
            iter += 1
            if iter == max_iter:
                print("up to max iter", max_iter)
                break
    v_bool_array = tmp_v.A > 0
    # print(tmp_v.A.T)

    return all_index_list, v_bool_array


def get_w_adjust_normal_list(adata, all_index_list):
    n = adata.shape[0]
    w = adata.obsp["connectivities"]
    w_adjust_normal_list = []

    for tmp_index_list in all_index_list:
        # print(tmp_index_list)
        w_adjust = csr_matrix(w.shape)  # 调整的权重矩阵
        inv_d_adjust = csr_matrix(w.shape)  # 调整的权重矩阵对应度矩阵求逆

        w_adjust[:, tmp_index_list] = w[:, tmp_index_list]  # 没有恢复过的细胞进行聚合
        w_adjust[tmp_index_list, :] = 0  # 恢复过的细胞保持不变
        w_adjust[tmp_index_list, tmp_index_list] = 1
        # print(w_adjust.A)

        inv_degree_array = 1 / w_adjust.sum(axis=1).A
        inv_d_adjust[range(n), range(n)] = inv_degree_array  # 度矩阵取逆
        # print(inv_d_adjust.A)

        w_adjust_normal = inv_d_adjust @ w_adjust  # 归一化的权重矩阵
        # print(w_adjust_normal.A)
        w_adjust_normal_list.append(w_adjust_normal)
        adata.uns["sample_recover"]["w_adjust_normal_list"] = w_adjust_normal_list

        # print("===============")

    return w_adjust_normal_list


def moment_obs_attribute(adata, subsample_adata, attribute, plot=True):
    # 为adata.obs赋新列
    adata.obs[attribute] = 0
    adata.obs.loc[subsample_adata.obs.index, attribute] = subsample_adata.obs[attribute]
    tmp_v = adata.obs[attribute]
    tmp_v = csr_matrix(tmp_v).T
    # 执行平滑
    for w_adjust_normal in adata.uns["sample_recover"]["w_adjust_normal_list"]:
        if plot:
            sc.pl.umap(adata, color=attribute)
        tmp_v = w_adjust_normal@tmp_v
        adata.obs[attribute] = tmp_v.A
    if plot:
        sc.pl.umap(adata, color=attribute) # 最后一次平滑结果


def moment_layer_attribute(adata, subsample_adata, attribute="velocity"):
    adata.layers[attribute] = np.zeros(adata.shape)
    adata.layers[attribute][adata.uns["sample_recover"]["index_list"]] = subsample_adata.layers[attribute]
    tmp_v = adata.layers[attribute]
    tmp_v = csr_matrix(tmp_v)
    # 执行平滑
    for w_adjust_normal in adata.uns["sample_recover"]["w_adjust_normal_list"]:
        tmp_v = w_adjust_normal@tmp_v
        adata.layers[attribute] = tmp_v
    if attribute=="velocity":
        adata.layers[attribute] = adata.layers[attribute].A


def moment_obsm_attribute(adata, subsample_adata, attribute="velocity_umap", plot=True):
    adata.obsm[attribute] = np.zeros((adata.shape[0], subsample_adata.obsm[attribute].shape[1]))
    adata.obsm[attribute][adata.uns["sample_recover"]["index_list"]] = subsample_adata.obsm[attribute]
    tmp_v = adata.obsm[attribute]
    tmp_v = csr_matrix(tmp_v)
    # 执行平滑
    for w_adjust_normal in adata.uns["sample_recover"]["w_adjust_normal_list"]:
        if attribute == "velocity_umap" and plot:
            scv.pl.velocity_embedding(adata, basis="umap")
        tmp_v = w_adjust_normal@tmp_v
        adata.obsm[attribute] = tmp_v.A
    if plot:
        scv.pl.velocity_embedding(adata, basis="umap") # 最后一次平滑结果


def write_adata(adata, dirname="tmp"):
    if not os.path.exists(dirname):
        os.mkdir(dirname)
        print("create %s" % dirname)
        if "sample_recover" in adata.uns.keys():
            # 需要单独保存uns中的sample_recover
            sample_recover_pkl_filename = "%s/sample_recover.pkl" % dirname
            with open(sample_recover_pkl_filename, "wb") as f:
                pickle.dump(adata.uns["sample_recover"], f)
            del adata.uns["sample_recover"]
            print("save %s" % sample_recover_pkl_filename)
        adata_filename = "%s/adata.h5ad" % dirname
        adata.write(adata_filename)
        print("save %s" % adata_filename)
    else:
        print("%s exist!" % dirname)


def read_adata(dirname="tmp"):
    if os.path.exists(dirname):
        adata_filename = "%s/adata.h5ad" % dirname
        adata = sc.read_h5ad(adata_filename)
        print("load %s" % adata_filename)
        if "sample_recover.pkl" in os.listdir(dirname):
            # 需要单独读取uns中的sample_recover
            sample_recover_pkl_filename = "%s/sample_recover.pkl" % dirname
            with open(sample_recover_pkl_filename, "rb") as f:
                adata.uns["sample_recover"] = pickle.load(f)
            print("load %s" % sample_recover_pkl_filename)
        return adata
    else:
        print("%s not exist!" % dirname)


########################################################################################################################
#######################################################邻居构建相关#######################################################
# 分别提取批次内部、批次间的mask
def get_mask(array, batch_pair_list):
    sorted_array_index_in_array = np.argsort(array) # 在array的排序下标，方便后续转换
    array_index_in_sorted_array = np.argsort(sorted_array_index_in_array)
    # value_list, count_list = np.unique(array, return_counts=True) # 这里由于排序问题，下标可能会混乱
    value_count_dict = dict(array.value_counts())
    value_list = list(array.cat.categories)
    count_list = [value_count_dict[value] for value in value_list] # 写的屎山，只能这样弥补了

    start_pos_list = [0] + [np.sum(count_list[:i+1]) for i in range(len(value_list)-1)]
    end_pos_list = np.array(start_pos_list) + np.array(count_list) # 注意这里是对应位置相加，需要转化为array
    pos_dict = dict(zip(value_list, zip(start_pos_list, end_pos_list))) # 各个batch的区间范围

    batch_dict = {} # 各个batch对整理成字典格式
    for value in value_list:
        batch_dict[value] = []
    for source, target in batch_pair_list:
        if target not in batch_dict[source]:
            batch_dict[source].append(target)
        if source not in batch_dict[target]:
            batch_dict[target].append(source)

    n = len(array)
    bnn_mask = np.zeros((n,n))
    knn_mask = np.zeros((n,n))
    for value in value_list:
        row_start = pos_dict[value][0]
        row_end = pos_dict[value][1]
        knn_mask[row_start:row_end,:][:,row_start:row_end] = True # 批内邻居
        for target_batch in batch_dict[value]:
            col_start = pos_dict[target_batch][0]
            col_end = pos_dict[target_batch][1]
            bnn_mask[row_start:row_end,:][:,col_start:col_end] = True

    knn_mask = knn_mask[array_index_in_sorted_array, :][:, array_index_in_sorted_array]
    bnn_mask = bnn_mask[array_index_in_sorted_array, :][:, array_index_in_sorted_array]

    return knn_mask, bnn_mask


# 构建bnn
def neighbor(adata,
             n_knn_neighbors=3,
             n_bnn_neighbors=3,
             batch_key="batch",
             batch_pair_list=None,
             mnn=True,
             n_pcs=None,
             use_rep=None,
             use_highly_variable=True,
             metric="euclidean"
             ):
    n = adata.shape[0]
    # TODO: 选择计算的矩阵
    use_rep = _get_rep(adata=adata, use_rep=use_rep, n_pcs=n_pcs)
    if use_rep == "X_pca":
        _set_pca(adata=adata, n_pcs=n_pcs, use_highly_variable=use_highly_variable)

        n_duplicate_cells = len(get_duplicate_cells(adata))
        if n_duplicate_cells > 0:
            logg.warn(
                f"You seem to have {n_duplicate_cells} duplicate cells in your data.",
                "Consider removing these via pp.remove_duplicate_cells.",
            )
    X = adata.X if use_rep == "X" else adata.obsm[use_rep]
    # print(use_rep)
    # print(X.shape)
    batch_series = adata.obs[batch_key]

    # TODO: 批次名，批次对需要以参数指定或在这里顺序生成或全排列
    if batch_pair_list == None:
        batch_list = list(batch_series.cat.categories)
        # batch_pair_list = [[batch_list[i], batch_list[i+1]] for i in range(len(batch_list)-1)] # 顺序
        # 全排列
        batch_pair_list = []
        l = len(batch_list)
        for i in range(l):
            for j in range(i + 1, l):
                batch_pair_list.append([batch_list[i], batch_list[j]])
    logg.info(f"pair_list : {batch_pair_list}", r=True)
    # print("pair_list", batch_pair_list)

    # TODO: mask矩阵构造
    knn_mask, bnn_mask = get_mask(batch_series, batch_pair_list)

    # 利用mask矩阵在全距离矩阵上获得BNN和KNN需要矩阵
    distances = pairwise_distances(X, metric=metric)  # 整体构造距离矩阵
    knn_distances_array = np.where(knn_mask == 1, distances, np.inf)
    bnn_distances_array = np.where(bnn_mask == 1, distances, np.inf)

    # np.savetxt("knn_distances.csv", knn_distances_array, delimiter=",")
    # np.savetxt("bnn_distances.csv", bnn_distances_array, delimiter=",")

    # 提取n*nn_knn_neighbors的距离array
    knn_indices, knn_nn_distances = _get_indices_distances_from_dense_matrix(knn_distances_array, n_knn_neighbors)
    bnn_indices, bnn_nn_distances = _get_indices_distances_from_dense_matrix(bnn_distances_array, n_bnn_neighbors)
    # 转化为n*n距离矩阵
    knn_distances = _get_sparse_matrix_from_indices_distances_numpy(knn_indices, knn_nn_distances, n, n_knn_neighbors)
    bnn_distances = _get_sparse_matrix_from_indices_distances_numpy(bnn_indices, bnn_nn_distances, n, n_bnn_neighbors)
    
    # TODO: 这里计算knn时，也要提前过滤
    for i in range(n):
            source_id = i
            for j in range(n_knn_neighbors):
                target_id = knn_indices[source_id][j]
                if knn_mask[source_id][target_id]==1.0:
                    pass
                else:
                    knn_nn_distances[i][j] = 0
                    knn_indices[i][j] = -1
    # 计算n*n的权重矩阵
    knn_distances, knn_connectivities = _compute_connectivities_umap(knn_indices, knn_nn_distances, n, n_knn_neighbors)
    # bnn_distances, bnn_connectivities = _compute_connectivities_umap(bnn_indices, bnn_nn_distances, n, n_bnn_neighbors) # 这里的计算没什么用
    # TODO: 批次间使用互近邻
    if mnn:
        # 批次间的邻居使用互近邻
        mnn_nn_distances = np.zeros(bnn_nn_distances.shape)
        mnn_indices = bnn_indices.copy()
        # 过滤只保留互近邻
        for i in range(n):
            source_id = i
            for j in range(n_bnn_neighbors):
                target_id = mnn_indices[source_id][j]
                # if source_id in mnn_indices[target_id]
                # 这里发现批次间的邻居出错了，多加bnn_mask的判断
                if source_id in mnn_indices[target_id] and bnn_mask[source_id][target_id]==1.0:
                    mnn_nn_distances[i][j] = bnn_nn_distances[i][j]
                else:
                    mnn_nn_distances[i][j] = 0
                    mnn_indices[i][j] = -1
        mnn_distances, mnn_connectivities = _compute_connectivities_umap(mnn_indices, mnn_nn_distances, n, n_bnn_neighbors)
        bnn_indices = mnn_indices
        bnn_distances = mnn_distances
        bnn_connectivities = mnn_connectivities

    # 拼接
    indices = np.concatenate([knn_indices, bnn_indices], axis=1)
    distances = knn_distances + bnn_distances
    connectivities = knn_connectivities + bnn_connectivities

    # 构造Neighbor对象
    class Neighbor():
        def __init__(self, indices, distances, connectivities):
            self.knn_indices = indices
            self.distances = distances
            self.connectivities = connectivities

    neighbors = Neighbor(indices, distances, connectivities)

    # 传入adata，这里的一些参后期还需要微调
    _set_neighbors_data(adata,
                        neighbors,
                        n_neighbors=n_knn_neighbors + n_bnn_neighbors,
                        method="bnn",
                        metric=metric,
                        n_pcs=n_pcs,
                        use_rep=use_rep,
                        )
    return knn_mask, bnn_mask


# 预处理
def preprocess(adata, n_bnn_neighbors=15, n_knn_neighbors=15, batch_mode="batch", batch_key="batch", batch_pair_list=None, sample_mode=None):
    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
    if batch_mode == "batch":
        # 批次间单独建立邻居
        logg.info("calculating knn and bnn mask...")
        knn_mask, bnn_mask = neighbor(adata, n_bnn_neighbors=n_bnn_neighbors, n_knn_neighbors=n_knn_neighbors, batch_key=batch_key, batch_pair_list=batch_pair_list)
        logg.info("smoothing...")
    else:
        logg.info("using scvelo neighbors...")
    scv.pp.moments(adata, n_pcs=30, n_neighbors=n_bnn_neighbors + n_knn_neighbors)
    if sample_mode == None:
        return knn_mask, bnn_mask
    else:
        # 执行抽样
        subsample_adata, index_list = sample(adata, mode=sample_mode)
        # 计算相关矩阵
        all_index_list, v_bool_array = get_all_index_list(adata, index_list)
        w_adjust_normal_list = get_w_adjust_normal_list(adata, all_index_list)
        return knn_mask, bnn_mask, subsample_adata


########################################################################################################################
######################################################谱系亚群相关########################################################
def get_adata_velocity_similarity(adata, cluster_name, cluster_key="clusters"):
    from sklearn.metrics.pairwise import cosine_similarity
    velocity_array = adata.obsm["velocity_umap"][adata.obs[cluster_key] == cluster_name]  # 此处较之前做了修改，提取特定簇的速率矩阵
    avg_velocity_array = velocity_array.mean(axis=0)  # 该聚类的平均速率向量
    avg_velocity_array = avg_velocity_array[np.newaxis, :].repeat(velocity_array.shape[0], axis=0)  # 向量转化为矩阵，方便后续运算

    similarity_array = cosine_similarity(velocity_array, avg_velocity_array).diagonal()  # 相似性矩阵对角线元素即为对应元素的相似性
    avg_similarity = similarity_array.mean()
    return avg_similarity


# 测试一下所有聚类的相似性值
def test_cluster_velocity_similarity(adata, cluster_key="clusters"):
    similarity_dict = {}
    for cluster_name in adata.obs[cluster_key].unique():
        similarity = get_adata_velocity_similarity(adata, cluster_name)
        similarity_dict[cluster_name] = similarity
    return similarity_dict


def get_sub_recluster(adata, cluster_name, k=2, sub_recluster_key="sub_recluster", cluster_key="clusters",
                      alpha=1.2, beta=0.8):
    from sklearn.preprocessing import StandardScaler, MinMaxScaler  # 预处理时特征归一化
    from sklearn.cluster import KMeans  # KMeans聚类
    from sklearn.metrics import silhouette_score  # 轮廓系数指标

    # 创建adata.obs新列，关注某群
    bool_series = adata.obs[cluster_key] == cluster_name
    adata.obs[sub_recluster_key] = np.where(bool_series, 1, 0)  # 创建新的列，用于区分亚型内部聚类，非0值为关注的聚类
    sub_index2index = dict(list(enumerate(np.where(bool_series)[0])))  # 关注某群中的下表到adata中的下标映射

    # 特征拼接
    reduction_visualization_features = adata.obsm["X_umap"][bool_series]  # 降维可视化特征
    velocity_features = adata.obsm["velocity_umap"][bool_series]  # 低维速率特征
    merged_fetureas = np.concatenate([reduction_visualization_features, velocity_features], axis=1)  # 特征拼接
    # TODO: 此处特征归一化预处理
    scaler = StandardScaler().fit(merged_fetureas)
    # scaler = MinMaxScaler().fit(merged_fetureas) # 用最大最小归一化替换
    merged_fetureas = scaler.transform(merged_fetureas)

    # kmeans聚类
    kmeans = KMeans(n_clusters=k, random_state=0).fit(merged_fetureas)

    # 保存到原本的adata.obs新列上
    sub_index = sub_index2index.keys()
    index = list(map(lambda x: sub_index2index[x], sub_index))
    adata.obs[sub_recluster_key][index] = kmeans.labels_ + 1

    # 轮廓系数指标计算，作为k选择的参考
    # TODO: 此处评价的时候，进行加权评价
    d = int(merged_fetureas.shape[1] / 2)
    adjust_merged_fetureas = np.concatenate([alpha * merged_fetureas[:, :d], beta * merged_fetureas[:, d:]], axis=1)

    return silhouette_score(adjust_merged_fetureas, kmeans.labels_)


# 对特定簇尝试多个k寻找其谱系亚群
def test_k_sub_recluster(adata, cluster_name, max_k=5, plot=True, visualization_mode="global"):
    max_score = 0
    max_score_k = 2
    for k in range(2, max_k + 1):
        sub_recluster_key = "sub_recluster(key=%d)" % k
        score = get_sub_recluster(adata, cluster_name, k=k, sub_recluster_key=sub_recluster_key)  # 计算指标
        if score > max_score:
            # 记录指标最大的k值
            max_score = score
            max_score_k = k
        print("k=%d, score=%.3f" % (k, score))
        if plot:
            # 颜色控制
            palette = sns.color_palette()
            if adata.shape[0] > 100:
                arrow_length = 8
            else:
                arrow_length = 1
            if visualization_mode == "global":
                scv.pl.velocity_embedding(adata, color=sub_recluster_key,
                                          palette=palette,
                                          arrow_length=arrow_length,
                                          dpi=300)  # 全局可视化
            else:
                scv.pl.velocity_embedding(adata[adata.obs["clusters"] == cluster_name], basis="umap",
                                          color=sub_recluster_key,
                                          arrow_length=arrow_length,
                                          dpi=300)  # 当前簇的可视化
    return max_score_k


def get_adjust_similarity_score(adata, cluster_key="clusters"):
    similarity_score_dict = test_cluster_velocity_similarity(adata, cluster_key=cluster_key)
    icvcoh_score_dict = inner_cluster_coh(adata, k_cluster=cluster_key, k_velocity="velocity")[0]
    mid_icvcoh_score = np.median(list(icvcoh_score_dict.values()))
    mid_similarity_score = np.median(list(similarity_score_dict.values()))
    k_list = similarity_score_dict.keys()
    score_dict = {}
    for k in k_list:
        if icvcoh_score_dict[k] > mid_icvcoh_score and similarity_score_dict[k] < mid_similarity_score:
            # icvcoh_score高于平均值，而similarity_score低于平均值才算过了门槛，参与计算
            score = (1 - similarity_score_dict[k]) * icvcoh_score_dict[k]
            score_dict[k] = score
    return score_dict


########################################################################################################################
###########################################################其他##########################################################
# 训练好的模型结果保存到adata
def add_velovi_outputs_to_adata(adata, vae):
    latent_time = vae.get_latent_time(n_samples=25)
    velocities = vae.get_velocity(n_samples=25, velo_statistic="mean")

    t = latent_time
    scaling = 20 / t.max(0)

    adata.layers["velocity"] = velocities / scaling
    adata.layers["latent_time_velovi"] = latent_time

    adata.var["fit_alpha"] = vae.get_rates()["alpha"] / scaling
    adata.var["fit_beta"] = vae.get_rates()["beta"] / scaling
    adata.var["fit_gamma"] = vae.get_rates()["gamma"] / scaling
    adata.var["fit_t_"] = (
        torch.nn.functional.softplus(vae.module.switch_time_unconstr)
        .detach()
        .cpu()
        .numpy()
    ) * scaling
    adata.layers["fit_t"] = latent_time.values * scaling[np.newaxis, :]
    adata.var['fit_scaling'] = 1.0
