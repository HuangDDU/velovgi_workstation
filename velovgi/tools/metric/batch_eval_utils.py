import numpy as np

from .eval_utils import  keep_type
from sklearn.metrics.pairwise import cosine_similarity

def keep_different_batch(adata, nodes, batch, k_batch="batch"):
    if len(nodes) > 0:
        return list(nodes[adata.obs[k_batch][nodes].values != batch])
    else:
        return []


def batch_cross_boundary_correctness(
        adata,
        k_cluster,
        k_velocity,
        cluster_edges,
        k_batch="batch",
        return_raw=False,
        x_emb="X_umap"
):
    """Cross-Boundary Direction Correctness Score (A->B)

    Args:
        adata (Anndata):
            Anndata object.
        k_cluster (str):
            key to the cluster column in adata.obs DataFrame.
        k_velocity (str):
            key to the velocity matrix in adata.obsm.
        cluster_edges (list of tuples("A", "B")):
            pairs of clusters has transition direction A->B
        return_raw (bool):
            return aggregated or raw scores.
        x_emb (str):
            key to x embedding for visualization.

    Returns:
        dict:
            all_scores indexed by cluster_edges or mean scores indexed by cluster_edges
        float:
            averaged score over all cells.

    """
    scores = {}
    all_scores = {}

    x_emb = adata.obsm[x_emb]
    if x_emb == "X_umap":
        v_emb = adata.obsm['{}_umap'.format(k_velocity)]
    else:
        v_emb = adata.obsm[[key for key in adata.obsm if key.startswith(k_velocity)][0]]

    for u, v in cluster_edges:
        sel = adata.obs[k_cluster] == u
        nbs = adata.uns['neighbors']['indices'][sel]  # [n * 30]

        boundary_nodes = map(lambda nodes: keep_type(adata, nodes, v, k_cluster), nbs)
        # TODO: 只保留不同批次间的邻居关系
        boundary_nodes = list(boundary_nodes)
        # print("整体", (u, v), [list(i) for i in boundary_nodes])
        different_batch_boundary_nodes = []
        index = adata.obs[sel].index
        for i in range(len(index)):
            batch = adata.obs[k_batch][index[i]]
            different_batch_boundary_nodes.append(keep_different_batch(adata, boundary_nodes[i], batch, k_batch=k_batch))
        different_batch_boundary_nodes = list(different_batch_boundary_nodes)  # map只能转换一次，这里为了方便查看
        # print("批次", (u, v), different_batch_boundary_nodes)

        x_points = x_emb[sel]
        x_velocities = v_emb[sel]

        type_score = []
        for x_pos, x_vel, nodes in zip(x_points, x_velocities, different_batch_boundary_nodes):
            if len(nodes) == 0:
                continue
            position_dif = x_emb[nodes] - x_pos
            dir_scores = cosine_similarity(position_dif, x_vel.reshape(1, -1)).flatten()
            type_score.append(np.mean(dir_scores))

        scores[(u, v)] = np.mean(type_score)
        all_scores[(u, v)] = type_score

    if return_raw:
        return all_scores

    return scores, np.mean([sc for sc in scores.values()])


def batch_inner_cluster_coh(adata, k_cluster, k_velocity, k_batch="batch", return_raw=False):
    """In-cluster Coherence Score.

    Args:
        adata (Anndata):
            Anndata object.
        k_cluster (str):
            key to the cluster column in adata.obs DataFrame.
        k_velocity (str):
            key to the velocity matrix in adata.obsm.
        return_raw (bool):
            return aggregated or raw scores.

    Returns:
        dict:
            all_scores indexed by cluster_edges mean scores indexed by cluster_edges
        float:
            averaged score over all cells.

    """
    clusters = np.unique(adata.obs[k_cluster])
    scores = {}
    all_scores = {}

    for cat in clusters:
        sel = adata.obs[k_cluster] == cat
        nbs = adata.uns['neighbors']['indices'][sel]
        same_cat_nodes = map(lambda nodes: keep_type(adata, nodes, cat, k_cluster), nbs)

        # TODO: 只保留不同批次间的邻居关系
        same_cat_nodes = list(same_cat_nodes)
        # print("整体" ,cat, [list(i) for i in same_cat_nodes])
        different_batch_same_cat_nodes = []
        index = adata.obs[sel].index
        for i in range(len(index)):
            batch = adata.obs[k_batch][index[i]]
            different_batch_same_cat_nodes.append(keep_different_batch(adata, same_cat_nodes[i], batch, k_batch=k_batch))
        different_batch_same_cat_nodes = list(different_batch_same_cat_nodes)  # map只能转换一次，这里为了方便查看
        # print("批次" ,cat, different_batch_same_cat_nodes)

        velocities = adata.layers[k_velocity]
        cat_vels = velocities[sel]
        cat_score = [cosine_similarity(cat_vels[[ith]], velocities[nodes]).mean()
                     for ith, nodes in enumerate(different_batch_same_cat_nodes)
                     if len(nodes) > 0]
        all_scores[cat] = cat_score
        scores[cat] = np.mean(cat_score)

    if return_raw:
        return all_scores

    return scores, np.mean([sc for sc in scores.values()])