import numpy as np
import pandas as pd

from .eval_utils import cross_boundary_correctness, inner_cluster_coh, summary_scores
from .batch_eval_utils import batch_cross_boundary_correctness, batch_inner_cluster_coh


# 计算指标前的预处理
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



# 汇总计算指标
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


# 根据指标字典转化array
def get_result(exp_metrics):
    results = {}
    results["CBDir"] = np.concatenate([np.array(x) for x in exp_metrics["CBDir"].values()])
    results["ICVCoh"] = np.concatenate([np.array(x) for x in exp_metrics["ICVCoh"].values()])
    # TODO: 自己添加的指标以后再加上去
    # results["BCBDir"] = np.concatenate([np.array(x) for x in exp_metrics["BCBDir"].values()])
    # results["BICVCoh"] = np.concatenate([np.array(x) for x in exp_metrics["BICVCoh"].values()])
    # 空缺部分补为None
    max_len = max([len(results[key]) for key in results.keys()])
    for key in results.keys():
        if len(results[key]) < max_len:
            new_array = np.ones(max_len) * np.nan
            new_array[:len(results[key])] = results[key]
            results[key] = new_array
    return results


# 获得所有模型整体的指标DataFrame
def get_metric_total_df(model_names, adata_list, cluster_edges, cluster_key="clusters", batch_key=None, return_raw=True):
    dfs = []

    for tmp_adata in adata_list:
        adata_velo = pre_metric(tmp_adata, "velocity")
        exp_metrics = summary_metric(adata_velo, cluster_edges, k_cluster = cluster_key, k_batch = batch_key, return_raw=True)
        result = get_result(exp_metrics)
        dfs.append(pd.DataFrame(result))

    df_list = []
    for i in range(len(dfs)):
        tmp_df = dfs[i]

        df_1 = tmp_df[["CBDir"]] # 这样是为了方便后续的字符串列
        df_1["Metric"] = "CBDir"
        df_1["Score"] = tmp_df["CBDir"]

        df_2 = tmp_df[["ICVCoh"]]
        df_2["Metric"] = "ICVCoh"
        df_2["Score"] = tmp_df["ICVCoh"]

        # TODO: 自己添加的指标以后再加上去
        # df_3 = tmp_df[["BCBDir"]]
        # df_3["Metric"] = "BCBDir"
        # df_3["Score"] = tmp_df["BCBDir"]

        # df_4 = tmp_df[["BICVCoh"]]
        # df_4["Metric"] = "BICVCoh"
        # df_4["Score"] = tmp_df["BICVCoh"]

        # df_ = pd.concat([df_1, df_2, df_3, df_4], axis=0)
        df_ =  pd.concat([df_1, df_2], axis=0)
        df_["Model"] = model_names[i] # 指标结果名称遵循特定的规则：模型_数据
        df_list.append(df_)

    df = pd.concat(df_list, axis=0)
    return df
