import numpy as np
import torch
from torch_geometric import seed_everything

import anndata as ad
import scanpy as sc
import scanpy.external as sce

def add_velovi_outputs_to_adata(adata, vae):
    """Add velocity/rate/t from model to adata

    Args:
        adata (_type_): adata object
        vae (_type_): vae model
    """
    latent_time = vae.get_latent_time(n_samples=25)
    velocities = vae.get_velocity(n_samples=25, velo_statistic="mean")
    velocities_u = vae.get_velocity(n_samples=25, velo_statistic="mean", velo_mode="unspliced")  # TODO: 加入unsplcied的速率
    t = latent_time
    scaling = 20 / t.max(0)

    adata.layers["velocity"] = velocities / scaling
    adata.layers["velocity_u"] = velocities_u / scaling # TODO: 加入unsplcied的速率
    adata.layers["latent_time_velovgi"] = latent_time

    adata.var["fit_alpha"] = vae.get_rates()["alpha"] / scaling
    adata.var["fit_beta"] = vae.get_rates()["beta"] / scaling
    adata.var["fit_gamma"] = vae.get_rates()["gamma"] / scaling
    adata.var["fit_t_"] = (
        torch.nn.functional.softplus(vae.module.switch_time_unconstr)
        .detach()
        .cpu()
        .numpy()
    ) * scaling
    adata.layers["fit_t"] = latent_time.values * scaling[np.newaxis, :]
    adata.var['fit_scaling'] = 1.0

# def get_latent_umap(adata, model, cluster_key="clusters", batch_key="batch_key", latent_key="X_latent_umap", random_seed=0):
#     # 提取隐变量，构造AnnData
#     latent_representation = model.get_latent_representation(adata)
#     latent_adata = ad.AnnData(latent_representation)
#     if random_seed:
#         from torch_geometric import seed_everything
#         seed = 0
#         seed_everything(seed)
#     # 执行scanpy的umap
#     latent_adata.obsm["X_pca"] = latent_adata.X.copy()
#     sc.pp.neighbors(latent_adata)
#     sc.tl.umap(latent_adata)
#     # 保存结果
#     adata.obsm[latent_key] = latent_adata.obsm["X_umap"]
#     return adata

# def get_latent_umap(adata, model, latent_key="X_latent", latent_umap_key="X_latent_umap", random_seed=0):
#     # 提取隐变量，构造AnnData
#     latent_representation = model.get_latent_representation(adata)
#     latent_adata = ad.AnnData(latent_representation)
#     if random_seed:
#         seed_everything(random_seed)
#     # 执行scanpy的umap
#     latent_adata.obsm["X_pca"] = latent_adata.X.copy()
#     sc.pp.neighbors(latent_adata)
#     sc.tl.umap(latent_adata)
#     # 保存结果
#     adata.obsm[latent_key] = latent_representation
#     adata.obsm[latent_umap_key] = latent_adata.obsm["X_umap"]
#     return adata


# 向后兼容
def get_latent_umap(adata, model, latent_key="X_latent", latent_umap_key="X_latent_umap", random_seed=0):
    return get_latent_embedding(adata, model, embedding_method="umap",latent_key=latent_key, random_seed=random_seed)


# 指定方法对隐变量降维
def get_latent_embedding(adata, model, embedding_method="umap",latent_key="X_latent", random_seed=0):
    latent_embedding_key = "%s_%s"%(latent_key, embedding_method)
    # 提取隐变量，构造AnnData
    latent_representation = model.get_latent_representation(adata)
    latent_adata = ad.AnnData(latent_representation)
    if random_seed:
        seed_everything(random_seed)
    
    # 选择降维方法对隐变量降维
    if embedding_method=="umap":
        # 执行scanpy的umap
        latent_adata.obsm["X_pca"] = latent_adata.X.copy()
        sc.pp.neighbors(latent_adata)
        sc.tl.umap(latent_adata)
    elif embedding_method=="pca":
        sc.tl.pca(latent_adata)
    elif embedding_method=="tsne":
        sc.tl.tsne(latent_adata)
    elif embedding_method=="phate":
        sce.tl.phate(latent_adata, random_state=random_seed)
        
    # 保存结果
    adata.obsm[latent_key] = latent_representation
    adata.obsm[latent_embedding_key] = latent_adata.obsm["X_%s"%embedding_method]
    return adata

