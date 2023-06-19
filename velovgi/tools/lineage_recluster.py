import numpy as np
import seaborn as sns

import scvelo as scv


from .metric.eval_utils import inner_cluster_coh

def get_adata_velocity_similarity(adata, cluster_name, cluster_key="clusters", basis="umap"):
    from sklearn.metrics.pairwise import cosine_similarity
    velocity_array = adata.obsm["velocity_%s"%basis][adata.obs[cluster_key] == cluster_name]  # 此处较之前做了修改，提取特定簇的速率矩阵
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


def get_sub_recluster(adata, cluster_name, k=2, sub_recluster_key="sub_recluster", cluster_key="clusters", basis="umap",
                      alpha=1.2, beta=0.8):
    from sklearn.preprocessing import StandardScaler, MinMaxScaler  # 预处理时特征归一化
    from sklearn.cluster import KMeans  # KMeans聚类
    from sklearn.metrics import silhouette_score  # 轮廓系数指标

    # 创建adata.obs新列，关注某群
    bool_series = adata.obs[cluster_key] == cluster_name
    adata.obs[sub_recluster_key] = np.where(bool_series, 1, 0)  # 创建新的列，用于区分亚型内部聚类，非0值为关注的聚类
    sub_index2index = dict(list(enumerate(np.where(bool_series)[0])))  # 关注某群中的下表到adata中的下标映射

    # 特征拼接
    reduction_visualization_features = adata.obsm["X_%s"%basis][bool_series]  # 降维可视化特征
    velocity_features = adata.obsm["velocity_%s"%basis][bool_series]  # 低维速率特征
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
def test_k_sub_recluster(adata, cluster_name, max_k=5, plot=True, visualization_mode="global", basis="umap"):
    max_score = 0
    max_score_k = 2
    for k in range(2, max_k + 1):
        sub_recluster_key = "sub_recluster(key=%d)" % k
        score = get_sub_recluster(adata, cluster_name, k=k, sub_recluster_key=sub_recluster_key, basis=basis)  # 计算指标
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
                                          basis=basis,
                                          dpi=300)  # 全局可视化
            else:
                scv.pl.velocity_embedding(adata[adata.obs["clusters"] == cluster_name],
                                          color=sub_recluster_key,
                                          arrow_length=arrow_length,
                                          basis=basis,
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