# %% [markdown]
# # 1. velovgi速率估计

# %%
import sys
sys.path = ["../../.."] + sys.path # 切换到项目目录下

import scanpy as sc
import scvelo as scv
import velovgi

# %%
from torch_geometric import seed_everything

seed = 0
seed_everything(seed)

# %% [markdown]
# ## 1. 数据读入

# %%
# 红系成熟部分
# adata_filename = "./data/erythroid_lineage.h5ad"
# adata_filename = "/home/huang/PyCode/scRNA/data/Gastrulation/erythroid_lineage.h5ad" # 实验室服务器的数据路径
adata_filename = "/home/21031211625/Code/rna_velocity_expirement/data/Gastrulation/erythroid_lineage.h5ad" # 集群服务器的数据路径

adata = scv.read(adata_filename)
batch_key = "stage"
cluster_key = "celltype"
adata

# %% [markdown]
# ## 2. 预处理

# %%
batch_pair_list = [
    ["E7.0", "E7.25"],
    ["E7.25", "E7.5"],
    ["E7.5", "E7.75"],
    ["E7.75", "E8.0"],
    ["E8.0", "E8.25"],
    ["E8.25", "E8.5"],
]
knn_mask, bnn_mask, subsample_adata = velovgi.pp.preprocess(adata, sample_mode="random", batch_key=batch_key, batch_pair_list=batch_pair_list)

# %% [markdown]
# ## 3. 默认参数执行VELOVGI并恢复可视化

# %% [markdown]
# 1. 模型训练

# %%
from pytorch_lightning import loggers

max_epochs=500
# max_epochs=50 # 效果已经很不错了
name = "%d_epoch(max_epochs=%d)"%(max_epochs, max_epochs)
logger = loggers.TensorBoardLogger(save_dir="./log", name=name) # 构造日志文件
    
velovgi.tl.VELOVGI.setup_anndata(adata=subsample_adata, spliced_layer="Ms", unspliced_layer="Mu")
velovgi_model = velovgi.tl.VELOVGI(subsample_adata)
velovgi_model.train(max_epochs=max_epochs, logger=logger)

# %% [markdown]
# 2. 结果输出

# %%
velovgi.tl.add_velovi_outputs_to_adata(subsample_adata, velovgi_model) # 模型输出
velovgi.pp.moment_recover(adata, subsample_adata) # 恢复
subsample_adata, adata

# %% [markdown]
# 3. 可视化

# %%
scv.tl.velocity_graph(adata)
scv.pl.velocity_embedding(adata, color=cluster_key)
scv.pl.velocity_embedding_stream(adata, color=cluster_key)
adata

# %% [markdown]
# ## 4. 结果保存

# %% [markdown]
# 1. subsample_adata保存

# %%
subsample_adata.write("./data/subsample_adata.h5ad")

# %% [markdown]
# 2. adata对象保存

# %%
adata_dir = "./data/adata"
velovgi.tl.write_adata(adata, adata_dir)
# adata = velovgi.tl.read_adata(adata_dir)

# %% [markdown]
# 2. 模型保存

# %%
model_dir = "./model/%s"%name
velovgi_model.save(model_dir)


