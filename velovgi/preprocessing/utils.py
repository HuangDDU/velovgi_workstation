import scvelo as scv
from scvelo import logging as logg

from .batch_network import neighbor
from .sample_recover import sample, get_all_index_list, get_w_adjust_normal_list

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