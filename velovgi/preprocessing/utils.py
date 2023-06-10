import numpy as np

import anndata as ad
import scanpy as sc
import scvelo as scv
from scvelo import logging as logg

from .batch_network_deprecated import neighbor as neighbor_deprecated
from .batch_network import neighbor
from .sample_recover import sample, get_all_index_list, get_w_adjust_normal_list

# 预处理
def preprocess_deprecated(adata, n_bnn_neighbors=15, n_knn_neighbors=15, batch_mode="batch", batch_key="batch", batch_pair_list=None, sample_mode=None):
    # Preprocess, generate multi-batch network
    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
    if batch_mode == "batch":
        # 批次间单独建立邻居
        logg.info("calculating knn and bnn mask...")
        knn_mask, bnn_mask = neighbor_deprecated(adata, n_bnn_neighbors=n_bnn_neighbors, n_knn_neighbors=n_knn_neighbors, batch_key=batch_key, batch_pair_list=batch_pair_list)
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
    

def preprocess(adata, n_bnn_neighbors=15, n_knn_neighbors=15, batch_mode="batch", batch_key="batch", batch_pair_list=None, sample_mode="random", is_ot=True):

    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
    if batch_mode == "batch":
        # 批次间单独建立邻居
        batch_pair_list = neighbor(adata, n_bnn_neighbors=n_bnn_neighbors, n_knn_neighbors=n_knn_neighbors, batch_key=batch_key, batch_pair_list=batch_pair_list, is_ot=is_ot)
    else:
        logg.info("using scvelo neighbors...")
    scv.pp.moments(adata, n_pcs=30, n_neighbors=n_bnn_neighbors + n_knn_neighbors)
    if sample_mode == None:
        return 
    else:
        # 执行抽样
        subsample_adata, index_list = sample(adata, mode=sample_mode)
        # 计算相关矩阵
        all_index_list, v_bool_array = get_all_index_list(adata, index_list)
        w_adjust_normal_list = get_w_adjust_normal_list(adata, all_index_list)
        return subsample_adata


# review里的批次整合方法
def review_preprocess(adata, batch_key,
                    cluster_key,
                    save=None,
                    transform_method="scvelo",
                    batch_removal_method="scgen",
                    epochs=10):
    # review论文中分别对S，U批次矫正
    import scib # TODO: notebook中，这里导入scib会与scvelo冲突

    # 执行数据转换
    adata = transform_adata(adata, transform_method=transform_method)

    # 提出数据变化后的矩阵构造对象
    adata_spliced = ad.AnnData(adata.layers["spliced"], obs=adata.obs[[batch_key, cluster_key]])
    adata_unspliced = ad.AnnData(adata.layers["unspliced"], obs=adata.obs[[batch_key, cluster_key]])

    # 执行矫正
    if batch_removal_method=="scgen":
        corrected_adata_spliced = scib.ig.scgen(adata_spliced, batch=batch_key, cell_type=cluster_key, epochs=epochs)
        corrected_adata_unspliced = scib.ig.scgen(adata_unspliced, batch=batch_key, cell_type=cluster_key, epochs=epochs)
    else:
        pass
        # TODO: 其他矫正方法
        # corrected_adata_spliced = scib.ig.harmony(adata_spliced, batch=batch_key)
        # corrected_adata_unspliced = scib.ig.harmony(adata_unspliced, batch=batch_key)

    # 保存到adata里
    corrected_S = corrected_adata_spliced.X
    corrected_U = corrected_adata_unspliced.X
    adata.X = corrected_S
    adata.layers["spliced"] = corrected_S
    adata.layers["unspliced"] = corrected_U

    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)

    if save:
        adata.write(save)

    return adata


# LatentVelo里的批次整合方法
def latentvelo_preprocess(adata, batch_key,
                          cluster_key,
                          save=None,
                          transform_method="scvelo",
                          batch_removal_method="scgen",
                          epochs=10):
    # latentvelo论文中考虑S，U相对数量比例的批次矫正方法
    import scib # TODO: notebook中，这里导入scib会与scvelo冲突
    
    # 执行数据转换
    adata = transform_adata(adata, transform_method=transform_method)

    # 计算比率关系矩阵
    M = np.array(adata.layers["spliced"].todense() + adata.layers["unspliced"].todense())
    masked_M = (M>0)*M + (M==0)
    R = np.array(adata.layers["spliced"].todense())/masked_M

    adata_sum = ad.AnnData(M, obs=adata.obs[[batch_key, cluster_key]])

    # 执行矫正
    if batch_removal_method=="scgen":
        corrected_adata_sum = scib.ig.scgen(adata_sum, batch=batch_key, cell_type=cluster_key, epochs=epochs)
    else:
        pass
        # TODO: 其他矫正方法
        # corrected_adata_sum = scib.ig.harmony(adata_sum, batch=batch_key)

    # 恢复并保存到adata里
    corrected_M = corrected_adata_sum.X
    corrected_S = corrected_M * R
    corrected_U = corrected_M * (1-R)
    adata.X = corrected_S
    adata.layers["spliced"] = corrected_S
    adata.layers["unspliced"] = corrected_U

    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)

    if save:
        adata.write(save)

    return adata

# 数据转换
def transform_adata(adata, transform_method="scvelo"):
    if transform_method=="scanpy":
        # 使用scanpy分别预处理两个矩阵
        adata_spliced = ad.AnnData(adata.layers["spliced"])
        adata_spliced.var.index = adata.var.index
        adata_unspliced = ad.AnnData(adata.layers["unspliced"])
        adata_unspliced.var.index = adata.var.index

        sc.pp.normalize_total(adata_spliced)
        sc.pp.log1p(adata_spliced)
        sc.pp.normalize_total(adata_unspliced)
        sc.pp.log1p(adata_unspliced)
        
        sc.pp.highly_variable_genes(adata_spliced, n_top_genes=2000) # 以spliced的高变化基因为准
        hvg_gene_list = list(adata_spliced.var[adata_spliced.var.highly_variable].index) 

        adata = adata[:, hvg_gene_list]
        adata.layers["spliced"] = adata_spliced[:, hvg_gene_list].X
        adata.layers["unspliced"] = adata_unspliced[:, hvg_gene_list].X
        return adata
    else:
        # 直接调用scvelo的方法，就不用分开做了
        scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
    return adata # 引用参数指针变化了，这里必须返回
