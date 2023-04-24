import numpy as np

from .eval_utils import cross_boundary_correctness, inner_cluster_coh, summary_scores
from .batch_eval_utils import batch_cross_boundary_correctness, batch_inner_cluster_coh

def summary_metric(
    adata,
    cluster_edges,
    k_cluster,
    # k_batch="batch",
    k_batch=None, # 可选是否对批次间的指标来评价
    k_velocity="velocity",
    x_emb="X_umap",
    return_raw=False,
    verbose=True
):
    exp_metrics = {}
    exp_metrics["CBDir"] = cross_boundary_correctness(adata, k_cluster, k_velocity, cluster_edges, True, x_emb)
    exp_metrics["ICVCoh"] = inner_cluster_coh(adata, k_cluster, k_velocity, True)
    if not k_batch == None:
        exp_metrics["BCBDir"] = batch_cross_boundary_correctness(adata, k_cluster, k_velocity, cluster_edges, k_batch, True, x_emb)
        exp_metrics["BICVCoh"] = batch_inner_cluster_coh(adata, k_cluster, k_velocity, k_batch, True)

    if return_raw:
        return exp_metrics
    else:
        exp_metrics_item_summary = {}  # 指标下每一项的汇总
        exp_metrics_summary = {}  # 指标汇总
        exp_metrics_item_summary["CBDir"], exp_metrics_summary["CBDir"] = summary_scores(exp_metrics["CBDir"])
        exp_metrics_item_summary["ICVCoh"], exp_metrics_summary["ICVCoh"] = summary_scores(exp_metrics["ICVCoh"])
        if not k_batch == None:
            exp_metrics_item_summary["BCBDir"], exp_metrics_summary["BCBDir"] = summary_scores(exp_metrics["BCBDir"])
            exp_metrics_item_summary["BICVCoh"], exp_metrics_summary["BICVCoh"] = summary_scores(exp_metrics["BICVCoh"])
        return exp_metrics_item_summary, exp_metrics_summary


def pre_metric(
    adata,
    k_velocity="velocity",
    basis="umap"
):
    adata.layers[k_velocity][np.isnan(adata.layers[k_velocity]).all(axis=1)] = 0  # 没有高维速率则置为0
    k_velocity_obsm = "%s_%s"%(k_velocity, basis)
    adata.obsm[k_velocity_obsm][np.isnan(adata.obsm[k_velocity_obsm]).all(axis=1)] = 0  # 没有低维速率则置为0
    if "velocity_genes" in adata.var.columns:
        # 有速率基因的话，需要提出来
        adata_velo = adata[:, adata.var.loc[adata.var["velocity_genes"] == True].index]
        return adata_velo
    else:
        return adata

