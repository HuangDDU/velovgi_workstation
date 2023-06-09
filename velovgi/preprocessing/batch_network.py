import numpy as np
from scipy.sparse import csr_matrix, bmat
from sklearn.metrics import pairwise_distances

import ot
import torch

import scanpy as sc
from scvelo.preprocessing.neighbors import _get_rep, _set_pca, get_duplicate_cells
from scvelo import logging as logg


# M为转移矩阵时，值越小表示结点越相似，largest为True表示寻找最高概率转移的结点
# M为距离矩阵时，值越大表示结点越相似，largest为False表示寻找最近邻的结点
# mode为intersection表示寻找共同、互相近邻的邻居关系，为union表示寻找单方向近邻的邻居关系
def filter_M(M, k, largest=True, mode="intersection"):
    col_indexs = torch.topk(torch.Tensor(M), k, largest=largest, axis=-1)[-1].numpy()
    row_indexs = np.repeat(np.arange(M.shape[0]), repeats=k).reshape(M.shape[0], k)

    M_row_new = np.zeros(M.shape)
    M_row_new[row_indexs,col_indexs] = M[row_indexs,col_indexs]

    N = M.T
    col_indexs = torch.topk(torch.Tensor(N), k, largest=largest, axis=-1)[-1].numpy()
    row_indexs = np.repeat(np.arange(N.shape[0]), repeats=k).reshape(N.shape[0], k)

    N_row_new = np.zeros(N.shape)
    N_row_new[row_indexs,col_indexs] = N[row_indexs,col_indexs]

    if mode == "intersection":
        return np.minimum(M_row_new, N_row_new.T)
    else:
        return np.maximum(M_row_new, N_row_new.T)


def get_normed_mnn_connectivities(mnn_connectivities):
    # # 概率矩阵行列归一化
    # row_sums = np.sum(mnn_connectivities, axis=1)
    # row_normalization_factor = np.sqrt(row_sums)
    # mnn_connectivities = mnn_connectivities / row_normalization_factor[:, np.newaxis] # 行归一化
    # column_sums = np.sum(mnn_connectivities, axis=0)
    # column_normalization_factor = np.sqrt(column_sums)
    # mnn_connectivities = mnn_connectivities / column_normalization_factor # 列归一化
    # mnn_connectivities

    # # 分别做行、列归一化之后计算
    mnn_connectivities1 = mnn_connectivities / np.linalg.norm(mnn_connectivities, axis=1, ord=1, keepdims=True)
    mnn_connectivities2 = mnn_connectivities / np.linalg.norm(mnn_connectivities, axis=0, ord=1, keepdims=True)
    mnn_connectivities = (mnn_connectivities1 + mnn_connectivities2) - mnn_connectivities1*mnn_connectivities2

    mnn_connectivities = np.nan_to_num(mnn_connectivities, nan=0) # 空置替换为0
    
    return mnn_connectivities


def get_mnn_connectivities(mnn_distances):
    # 先对距离取倒数，相当于成为概率矩阵
    mnn_connectivities = np.zeros(mnn_distances.shape)
    nonzero_indices = np.where(mnn_distances != 0)
    mnn_connectivities[nonzero_indices] = 1 / mnn_distances[nonzero_indices]
    
    mnn_connectivities = get_normed_mnn_connectivities(mnn_connectivities)
    
    return mnn_connectivities


# 构建bnn
def neighbor(adata,
             n_knn_neighbors=3,
             n_bnn_neighbors=3,
             batch_key="batch",
             batch_pair_list=None,
             mnn=True, # 以后再说用这个
             is_ot=True,
             n_pcs=None,
             use_rep=None,
             use_highly_variable=True,
             metric="euclidean"
             ):
    # 开始确定用哪个数据
    use_rep = _get_rep(adata=adata, use_rep=use_rep, n_pcs=n_pcs)
    if use_rep == "X_pca":
        _set_pca(adata=adata, n_pcs=n_pcs, use_highly_variable=use_highly_variable)

        n_duplicate_cells = len(get_duplicate_cells(adata))
        if n_duplicate_cells > 0:
            logg.warn(
                f"You seem to have {n_duplicate_cells} duplicate cells in your data.",
                "Consider removing these via pp.remove_duplicate_cells.",
            )
    logg.info(f"use_rep : {use_rep}", r=True)
    # X = adata.X if use_rep == "X" else adata.obsm[use_rep]

    batch_list = list(adata.obs[batch_key].cat.categories)
    adata_list = [adata[adata.obs[batch_key]==batch].copy() for batch in batch_list]

    # 指定稀疏矩阵的位置稍后拼接
    m = len(batch_list)
    connectivities_list = [[None for j in range(m)]for i in range(m)] 
    distances_list = [[None for j in range(m)]for i in range(m)]

    # 各个批次内部的AnnData分别做KNN
    for i in range(len(batch_list)):
        tmp_adata = adata_list[i]
        sc.pp.neighbors(tmp_adata,
                        n_neighbors=n_knn_neighbors,
                        n_pcs=n_pcs,
                        use_rep=use_rep,
                        metric=metric
                        )
        connectivities_list[i][i]=adata_list[i].obsp["connectivities"]
        distances_list[i][i]=adata_list[i].obsp["distances"]

    # 各批次之间做(最优传输或直接距离)MNN
    if batch_pair_list == None:
        batch_pair_list = list(zip(batch_list[:-1], batch_list[1:]))
    logg.info(f"batch_pair_list : {batch_pair_list}", r=True)
    for batch_pair in batch_pair_list:
        batch1_index, batch2_index = batch_list.index(batch_pair[0]), batch_list.index(batch_pair[1])
        adata1, adata2 = adata_list[batch1_index], adata_list[batch2_index]
        # X, Y = adata1.X, adata2.X
        X = adata1.X if use_rep == "X" else adata1.obsm[use_rep]
        Y = adata2.X if use_rep == "X" else adata2.obsm[use_rep]
        if (n_bnn_neighbors > adata1.shape[0]) or (n_bnn_neighbors > adata2.shape[0]):
            k = min(adata1.shape[0], adata2.shape[0])
            logg.info(f"pair {batch_pair} cells not enough, k={k}", r=True)
        else:
            k = n_bnn_neighbors
        distances = pairwise_distances(X,Y)
        if is_ot == True:
            # 最优传输的最近邻
            a, b = np.ones((adata1.shape[0],)) / adata1.shape[0], np.ones((adata2.shape[0],)) / adata2.shape[0]
            connectivities = ot.emd(a, b, distances) # 最优传输计算获得转移概率
            filtered_connectivities = filter_M(connectivities, k, largest=True)
            mnn_distances = np.where(filtered_connectivities>0, distances, 0) # 过滤距离矩阵中的元素
            mnn_connectivities = get_normed_mnn_connectivities(filtered_connectivities) # 转移矩阵归一化
        else:
            # 直接距离的最近邻
            mnn_distances = filter_M(distances, k, largest=False) # 距离矩阵提取互相近邻
            mnn_connectivities = get_mnn_connectivities(mnn_distances) # 距离矩阵转化为邻接矩阵

        # print(mnn_distances, mnn_connectivities)

        connectivities_list[batch1_index][batch2_index] =  mnn_connectivities
        connectivities_list[batch2_index][batch1_index] = mnn_connectivities.T
        distances_list[batch1_index][batch2_index] = mnn_distances
        distances_list[batch2_index][batch1_index] = mnn_distances.T

    adata_concat = sc.concat(adata_list)
    adata_concat.obsp["connectivities"] = bmat(connectivities_list).A # 按照位置拼接稀疏矩阵， 后续需要变序号，所以这里转成numpy矩阵
    adata_concat.obsp["distances"] = bmat(distances_list).A
    adata_concat = adata_concat[adata.obs.index]
    adata.obsp["connectivities"] = csr_matrix(adata_concat.obsp["connectivities"])
    adata.obsp["distances"] = csr_matrix(adata_concat.obsp["distances"])

    return batch_pair_list


# 这里传入的是series获得mask
def get_mask(batch_series, batch_pair_list):

    batch_list = list(batch_series.cat.categories)
    num_list = [(batch_series==batch).sum() for batch in batch_list]
    batch_pair_list = list(zip(batch_list[:-1], batch_list[1:])) if batch_pair_list == None else batch_pair_list

    # 指定稀疏矩阵的位置稍后拼接
    m = len(batch_list)
    knn_mask_list = [[None for j in range(m)]for i in range(m)] 
    bnn_mask_list = [[None for j in range(m)]for i in range(m)]

    # 批次内部knn构造
    for i in range(len(num_list)):
        num = num_list[i]
        knn_mask_list[i][i] = np.ones((num, num))

    # 外部bnn构造
    for batch_pair in batch_pair_list:
        batch1_index, batch2_index = batch_list.index(batch_pair[0]), batch_list.index(batch_pair[1])
        num1, num2 = num_list[batch1_index], num_list[batch2_index]
        bnn_mask_list[batch1_index][batch2_index] = np.ones((num1, num2))
        bnn_mask_list[batch2_index][batch1_index] = np.ones((num2, num1))

    # 重新排序
    sorted_indices = np.argsort(batch_series.cat.codes) # 原本数组元素在排序后数组中的位置
    original_indices = np.argsort(sorted_indices) # 排序后数组元素在原本数组中的位置
    knn_mask = bmat(knn_mask_list).A[original_indices, :][:, original_indices]
    bnn_mask = bmat(bnn_mask_list).A[original_indices, :][:, original_indices]
    return knn_mask, bnn_mask