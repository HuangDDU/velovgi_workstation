import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

import scanpy as sc
import scvelo as scv

# 获得变化矩阵
def get_transform_matrix(p=0.5,q=0.5, theta=np.pi/6):
    # 横纵坐标伸缩之后，纵坐标旋转
    transform_matrix = [
        [p,0],
        [q*np.sin(theta), q*np.cos(theta)]
    ]
    return transform_matrix


# 获得变化后的批次嵌入矩阵
def get_tranformed_batch_embeeding(
    adata,
    transform_matrix,
    embedding_key="X_umap",
    cluster_key="clusters",
    batch_key="batch",
    sep = None,
):
    sep = 4 if sep==None else sep
    batch_list = list(adata.obs[batch_key].cat.categories)
    embedding_transformed_batch_key = "%s_transformed_batch"%embedding_key
    adata.obsm[embedding_transformed_batch_key] = np.zeros(adata.obsm[embedding_key].shape) # 准备新的批次可视化嵌入坐标位置
    for i in range(len(batch_list)):
        batch = batch_list[i]
        tmp_series = adata.obs[batch_key]==batch
        tmp_adata = adata[tmp_series]
        tmp_adata.obsm[embedding_transformed_batch_key] = tmp_adata.obsm[embedding_key] @ transform_matrix - [0, sep*i]
        adata.obsm[embedding_transformed_batch_key][tmp_series] = tmp_adata.obsm[embedding_transformed_batch_key]


# 绘制各个批次的层
def draw_batch_layer(
    adata,
    transform_matrix,
    embedding_key="X_umap",
    cluster_key="clusters",
    batch_key="batch",
    sep = None,
    ax=None
):
    # 批次分层的平行四边形
    sep = 4 if sep==None else sep
    max_x, max_y = adata.obsm[embedding_key].max(axis=0)
    min_x, min_y = adata.obsm[embedding_key].min(axis=0)
    xy = np.array([
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y],
        ]) # 平行四边形平面计算
    xy_transformed = xy @ transform_matrix # 旋转后的平行四边形

    batch_list = list(adata.obs[batch_key].cat.categories)
    ax = plt.subplot() if ax==None else ax
    for i in range(len(batch_list)):
        batch = batch_list[i]
        tmp_xy_transformed = xy_transformed - (0, sep*i)
        text_pos = (tmp_xy_transformed[0] + tmp_xy_transformed[-1]) / 2 - (1, 0) # TODO: 这里批次的文字说明位置还需要调整
        plt.text(text_pos[0], text_pos[1], batch, size=15) # 批次的文字说明绘制
        ax.fill(tmp_xy_transformed[:, 0], tmp_xy_transformed[:, 1], "#00000011") # 每个批次的平行四边形平面绘制
    
    return ax


# 使用netowrkX绘制邻居关系
def draw_edge(
    adata,
    embedding_key="X_umap",
    cluster_key="clusters",
    batch_key="batch",
    neighbor_key="connectivities"
):
    from scipy.sparse import coo_matrix

    embedding = np.array([tuple(i) for i in adata.obsm[embedding_key]]) 
    pos = dict(zip(range(adata.shape[0]), embedding))
    coo_m = coo_matrix(adata.obsp[neighbor_key])
    edges = list(zip(coo_m.row, coo_m.col)) # 邻居关系
    nodes = range(adata.shape[0])

    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    
    source = np.array(adata.obs[batch_key][np.array(edges)[:,0]])
    target = np.array(adata.obs[batch_key][np.array(edges)[:,1]])
    # edge_color_list = ["red" if i==True else "black" for i in (source != target)] # TODO: 这里颜色需要可以调整
    edge_color_list = ["grey" if i==True else "black" for i in (source != target)]
    style_list = ["-." if i==True else "-" for i in (source != target)]
    nx.draw_networkx_edges(G, pos, edgelist=edges,width=0.5 ,edge_color=edge_color_list, style=style_list)

# 自定义的速率绘制函数
def draw_velocity(
    adata,
    embedding_key="X_umap",
    cluster_key="clusters",
    velocity_embedding_key = "velocity_umap",
    ax=None
):
    cluster2color = dict(zip(adata.obs[cluster_key].cat.categories, adata.uns["%s_colors"%cluster_key]))
    node_color_list = list(adata.obs[cluster_key].apply(lambda x:cluster2color[x]))
    quiver_kwargs = {
        "scale": 0.5,
        "scale_units": "xy",
        "linewidth": 1,
        "headlength": 10,
        "headwidth": 10,
        "width": 0.005, # TODO: 箭头调粗
        "color": node_color_list
    }
    X, Y = adata.obsm[embedding_key][:, 0], adata.obsm[embedding_key][:, 1]
    U, V = adata.obsm[velocity_embedding_key][:, 0], adata.obsm[velocity_embedding_key][:, 1]
    ax = plt.subplot() if ax==None else ax
    ax.quiver(X, Y, U, V, **quiver_kwargs)


def draw_batch_layer_embedding(
    adata,
    embedding_key="X_umap",
    cluster_key="clusters",
    batch_key="batch",
    transform_matrix_params={},
    sep = 5,
    embedding_plot_params={},
    show_edge=False,
    show_velocity=False,
    ax=None,
):
    transform_matrix = get_transform_matrix(**transform_matrix_params) # 获得变化矩阵
    get_tranformed_batch_embeeding(
        adata,
        transform_matrix,
        embedding_key,
        cluster_key,
        batch_key,
        sep,
    ) # 获得变化后的批次嵌入矩阵
    ax = draw_batch_layer(
        adata,
        transform_matrix,
        embedding_key,
        cluster_key,
        batch_key,
        sep,
        ax=ax
    ) # 绘制各个批次的层
    
    embedding_transformed_batch_key = "%s_transformed_batch"%embedding_key
    if show_edge:
        draw_edge(adata, embedding_key=embedding_transformed_batch_key, cluster_key=cluster_key, batch_key=batch_key)
    if show_velocity:
        # 使用自己写的函数
        # draw_velocity(adata, embedding_key=embedding_transformed_batch_key, cluster_key=cluster_key, ax=ax) 
        # 使用scvelo进行变化
        velocity_key = "velocity_%s"%embedding_key[2:]
        embedding_transformed_batch_velocity_key = "velocity_%s"%embedding_transformed_batch_key[2:]
        adata.obsm[embedding_transformed_batch_velocity_key] = adata.obsm[velocity_key] @ transform_matrix # 速率箭头也旋转
        scv.pl.velocity_embedding(adata, color=cluster_key, basis=embedding_transformed_batch_key, ax=ax)

    sc.pl.embedding(adata, color=cluster_key, basis=embedding_transformed_batch_key, ax=ax, show=False, **embedding_plot_params) # 最后绘制散点
    
    return ax


def draw_batch_layer_embedding_3d(
        adata,
        embedding_key="X_umap",
        cluster_key="clusters",
        batch_key="batch",
        show=True,
        size=None,
        opacity=1,
        width=1000,
        height=800):
    # 构造数据
    df = pd.DataFrame()
    df.index = adata.obs.index
    df["embedding_x"] = adata.obsm[embedding_key][:, 0]
    df["embedding_y"] = adata.obsm[embedding_key][:, 1]
    df[batch_key] = adata.obs[batch_key]
    df[cluster_key] = adata.obs[cluster_key]
    if size==None:
        size = int(20000/df.shape[0])
        size = 1 if size < 1 else size # 也不能太小了
    df["size"] = size # 大小

    fig = px.scatter_3d(df,
                        x="embedding_x", y="embedding_y", z=batch_key,
                        color=cluster_key, hover_name=batch_key,
                        size="size", size_max=size, opacity=opacity,
                        color_discrete_map=dict(
                            zip(adata.obs[cluster_key].cat.categories, adata.uns["%s_colors" % cluster_key])),
                        category_orders={
                            cluster_key: list(df[cluster_key].cat.categories),
                            batch_key: list(
                                reversed(list(df[batch_key].cat.categories)))  # 批次顺序在纵轴上展示
                        }
                        )  # 此处绘制三维散点图

    fig.update_traces(marker=dict(line=dict(width=0))) # 取消边框
    fig.update_layout(
        width=width,
        height=height,
    )
    if show:
        # 展示
        fig.show()
    else:
        # 不展示，等待用户更新画布
        return fig