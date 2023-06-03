# %% [markdown]
# # base

# %% [markdown]
# ## 1. 模型导入

# %%
import sys
sys.path = ["../../../../"] + sys.path # 切换到项目目录下

import scanpy as sc
import scvelo as scv
import velovgi

from ray import tune, air
from ray.air import session

# %% [markdown]
# ## 2. 数据读取

# %%
adata = velovgi.tl.read_adata("../../../erythroid_lineage/data/adata/")
subsample_adata = scv.read("../../../erythroid_lineage/data/subsample_adata.h5ad") # 使用这个AnnData做训练

cluster_key = "celltype"
batch_key="stage"

adata, subsample_adata

# %%
cluster_edges = [
    ("Blood progenitors 1", "Blood progenitors 2"), 
    ("Blood progenitors 2", "Erythroid1"), 
    ("Erythroid1", "Erythroid2"), 
    ("Erythroid2", "Erythroid3")
    ] # 已知的细胞类型间的分化信息

# %% [markdown]
# ## 3. RayTune定义

# %% [markdown]
# 1. 目标函数

# %%
from pytorch_lightning import loggers
from torch_geometric import seed_everything

# TODO: 正交调参每次只传入单个超参数， 其他默认参数后续需要修改
def train_velovgi(config):
    # 提取超参数
    # 随机数种子，确保结果的可复现性
    random_seed = config.get("random_seed", 0)
    # 模型结构参数
    n_hidden = config.get("n_hidden", 256)
    n_latent = config.get("n_latent", 10)
    n_layers = config.get("n_layers", 1)
    # 训练参数
    num_neighbors = [config.get("num_neighbors", 8)]*n_layers # 每层的邻居采样个数
    max_epochs = config.get("max_epochs", 10)
    batch_size = config.get("batch_size", 64)
    max_kl_weight = config.get("max_kl_weight", 0.8)

    name = ""
    for k,v in config.items():
        name += "%s_%s,"%(k, v)
    name = name[:-1]

    seed_everything(random_seed)
    # 模型训练
    logger = loggers.TensorBoardLogger(save_dir="./log", name=name)
    velovgi.tl.VELOVGI.setup_anndata(adata=subsample_adata, spliced_layer="Ms", unspliced_layer="Mu")
    velovgi_model = velovgi.tl.VELOVGI(subsample_adata, n_hidden=n_hidden, n_latent=n_latent, n_layers=n_layers)
    velovgi_model.train(num_neighbors=num_neighbors, max_epochs=max_epochs, batch_size=batch_size, plan_kwargs={"max_kl_weight": max_kl_weight}, logger=logger)

    # 模型恢复
    velovgi.tl.add_velovi_outputs_to_adata(subsample_adata, velovgi_model) # 模型输出
    velovgi.pp.moment_recover(adata, subsample_adata) # 恢复

    # 速率计算
    scv.tl.velocity_graph(adata)
    scv.pl.velocity_embedding(adata, color=cluster_key)
    scv.pl.velocity_embedding_stream(adata, color=cluster_key, save="%s.png"%name)

    # 计算指标评价
    adata_velo = velovgi.tl.pre_metric(adata)
    exp_metrics = velovgi.tl.summary_metric(adata_velo, cluster_edges, cluster_key)[-1] # 计算指标汇总后的结果

    session.report({"CBDir": exp_metrics["CBDir"], "ICVCoh": exp_metrics["ICVCoh"]})


# %% [markdown]
# 2. 搜索空间（暂时都是默认参数）

# %% [markdown]
# 3. 执行调参，等待传入实验名称和搜索空间

# %%
from ray.tune.schedulers import ASHAScheduler

def hyperparameter_tuner(name, search_space):
    tuner = tune.Tuner(
        train_velovgi,
        tune_config=tune.TuneConfig(
            metric="CBDir",
            mode="max",
            scheduler=ASHAScheduler()
        ),
        run_config=air.RunConfig(
            local_dir="./results", # Trail内部具体输出结果在这里保存
            name=name # 开启调参的Tensorboard日志
        ),
        param_space=search_space,
    )

    results = tuner.fit()
    return results


