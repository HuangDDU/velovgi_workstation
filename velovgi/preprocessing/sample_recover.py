import numpy as np
from scipy.sparse import csr_matrix
import networkx as nx

import scanpy as sc
import scvelo as scv


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
    #
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
    """Moment layer attribute, such as velocity matrix

    Args:
        adata (_type_): _description_
        subsample_adata (_type_): _description_
        attribute (str, optional): _description_. Defaults to "velocity".
    """
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
