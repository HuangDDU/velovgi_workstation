import math
import numpy as np
import matplotlib.pyplot as plt

import scvelo as scv
from scvelo.pl.utils import default_size

from .utils import calc_fig


# 指定ax绘制单个基因
def draw_velocity_gene(tmp_adata, adata, gene, cluster_key="clusters", size=None, ax=None):
    kwargs = dict(ax=ax, show=False, frameon=False) # 公用参数
    scv.pl.velocity_embedding(tmp_adata, color="black", basis=gene, size=size, alpha=1 ,**kwargs) # 绘制抽样到的细胞黑色速率箭头
    scv.pl.scatter(adata, color=cluster_key, basis=gene, **kwargs) # 先绘制一层底色，所有细胞按照细胞类型给颜色
    adata_cluster_list = list(adata.obs[cluster_key].cat.categories)
    scv.pl.scatter(tmp_adata, color=cluster_key, basis=gene, size=size, add_outline=adata_cluster_list, **kwargs) # 对抽样到的细胞添加边界


# 绘制多个基因的速率图到一个画布上
def draw_velocity_gene_list(adata, gene_list, cluster_key="clusters", n=None, size=None, cols=5):
    np.random.seed(0)
    cell_num = adata.shape[0]
    if n == None:
        n = int(cell_num*0.1) # 采样的细胞个数
        n = n if n < 1000 else 1000 # 在1000处阶段
    if size == None:
        size = default_size(adata)*math.log(adata.shape[0]) # 箭头的缩放比例

    index_list = np.random.choice(np.arange(cell_num), size=n, replace=False)
    tmp_adata = adata[index_list, gene_list]

    # rows = math.ceil(len(gene_list)/cols)
    # item_figsize = (5, 4) # 单张图的figsize
    # if rows == 1:
    #     cols = len(gene_list)
    # figsize = (item_figsize[0]*cols, item_figsize[1]*rows) # 整体figsize
    # fig, ax = plt.subplots(rows, cols, figsize=figsize)
    # if type(ax) == np.ndarray:
    #     ax = ax.flatten()
    # else:
    #     ax = np.array(ax)

    item_figsize = (5, 4) # 单张图的figsize
    fig, ax = calc_fig(len(gene_list), cols, item_figsize=item_figsize)

    for i in range(len(gene_list)):
        gene = gene_list[i]
        draw_velocity_gene(tmp_adata, adata, gene, cluster_key=cluster_key, size=size, ax=ax[i])
    
    # 其余关闭窗口坐标
    for i in range(len(gene_list), len(ax)):
        ax[i].axis("off") 
