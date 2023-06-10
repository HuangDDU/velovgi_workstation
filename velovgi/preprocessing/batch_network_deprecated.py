import numpy as np

from sklearn.metrics import pairwise_distances
from scanpy.neighbors import _get_indices_distances_from_dense_matrix, _get_sparse_matrix_from_indices_distances_numpy, _compute_connectivities_umap
from scvelo.preprocessing.neighbors import _set_neighbors_data
from scvelo.preprocessing.neighbors import _get_rep, _set_pca, get_duplicate_cells
from scvelo import logging as logg


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