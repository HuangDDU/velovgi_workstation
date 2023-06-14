import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ..preprocessing.batch_network import get_mask
import scvelo as scv

# 绘制批次间相关性的函数
def draw_batch_correlation(adata, batch_key="batch", vmin=0.5, vmax=1):
    batch_list = list(adata.obs[batch_key].cat.categories)
    adata_list = []
    for batch in batch_list:
        tmp_adata = adata[adata.obs[batch_key]==batch]
        adata_list.append(tmp_adata)

    # 获得平均值表达矩阵
    avg_exp_df = pd.DataFrame(index = list(adata.var.index))
    for i in range(len(adata_list)):
        batch = batch_list[i] # 样本名称
        adata = adata_list[i] # 样本对象
        avg_exp_df[batch] = adata.X.toarray().mean(axis=0) # 所有细胞的平均表达
    # 利用平均表达值计算相关性矩阵
    correlation_matrix = np.corrcoef(avg_exp_df.T)
    df = pd.DataFrame(correlation_matrix, index= batch_list, columns = batch_list)
    # 绘图
    sns.heatmap(df, square=True, vmin=vmin, vmax=vmax, fmt=".3f",
                linewidth=0.5, annot=True, cmap="RdBu_r")
    
# 添加了子图绘制的函数
def draw_batch_circos_ax(adata, batch_key="batch", ticks_interval=None, return_matrix_df=False, ax=None):
    from pycirclize import Circos
    
    # 提取边的数量
    row_names = list(adata.obs[batch_key].cat.categories)
    col_names = row_names
    matrix_data = np.zeros((len(row_names), len(col_names)))
    connectivities = adata.obsp["connectivities"]
    for i in range(len(row_names)):
        for j in range(len(col_names)):
            # if i > j:
            #     continue
            row_name = row_names[i]
            col_name = col_names[j]
            row_bool_series = adata.obs[batch_key] == row_name
            col_bool_series = adata.obs[batch_key] == col_name
            matrix_data[i,j] = (connectivities[row_bool_series,:][:,col_bool_series]>0).sum()
    matrix_data = matrix_data.astype("int")
    # matrix_df = pd.DataFrame(matrix_data, index=row_names, columns=col_names)

    # 添加细胞数目
    row_names = ["%s(%d)"%(row_name, (adata.obs[batch_key]==row_name).sum()) for row_name in row_names]
    col_names = row_names

    # 只保留上半部分矩阵
    adjust_matrix_data = np.zeros(matrix_data.shape)
    for i in range(matrix_data.shape[0]):
        for j in range(matrix_data.shape[0]):
            if i <= j:
                adjust_matrix_data[i][j] = matrix_data[i][j]
    adjust_matrix_df = pd.DataFrame(adjust_matrix_data, index=row_names, columns=col_names)

    # TODO: 这里自动提取计算ticks_interval
    if ticks_interval == None:
        base = 100 # 精确到指定位数上
        ticks_interval = int(adjust_matrix_df.sum().sum()/10/base)*base

    # 绘制
    circos = Circos.initialize_from_matrix(
        adjust_matrix_df,
        space=3,
        r_lim=(93, 100),
        cmap=dict(zip(row_names, adata.uns["%s_colors"%batch_key])),
        ticks_interval=ticks_interval,
        ticks_kws=dict(label_size=10),
        label_kws=dict(r=94, size=12, color="black"), # 暂时为了看清，只能设置为黑色了
    )

    fig = circos.plotfig(ax=ax)

    if return_matrix_df==True:
        return adjust_matrix_df
    
# 计算两种邻居个数的函数
def draw_batch_nn_umap(adata, batch_key="batch", cluster_key="clusters", title="batch_nn_umap", save=None):
    array = adata.obs[batch_key]
    batch_list = list(adata.obs[batch_key].cat.categories)
    n = len(batch_list)
    batch_pair_list = []
    for i in range(n):
        for j in range(i,n):
            batch_pair_list.append([batch_list[i], batch_list[j]])
    knn_mask, bnn_mask = get_mask(array, batch_pair_list)
    # (knn_mask.T == knn_mask).any(), (bnn_mask.T == bnn_mask).any() # 两个mask矩阵都是对称的
    connectivities = adata.obsp["connectivities"]
    # print((connectivities.A.T == connectivities.A).all()) # 对称的邻居图
    # 提取两种邻居数目
    adata.obs["knn"] = (np.where(knn_mask == 1, connectivities.A, 0) > 0).sum(axis=0)
    adata.obs["bnn"] = (np.where(bnn_mask == 1, connectivities.A, 0) > 0).sum(axis=0)

    adata.obs["knn"] = adata.obs["knn"].astype("float")
    adata.obs["bnn"] = adata.obs["bnn"].astype("float")
    # scv.pl.umap(adata, color=["knn","bnn"])

    cols = 5  # 每行的umap图个数，最好是奇数
    n_batch_list = len(batch_list)
    cols = min(cols, n_batch_list)
    rows_per_group = int(np.ceil(n_batch_list / cols))
    cols_per_big_umap = int((cols+1)/2)
    rows = 2*rows_per_group + cols_per_big_umap  # 前面是小umap图，后面是大umap图
    figsize = ((cols+1)*4, rows*4)
    fig, ax = plt.subplots(rows,
                           cols+1,
                           num=title,
                           figsize=figsize,
                           tight_layout=True)
    # plt.ioff() # 暂时不直接显示，作为结果返回，这种延迟展示会有问题，之后一股脑地显示出来
    fig.suptitle(title, fontsize=30)

    #################################################### 整体UMAP部分####################################################
    tmp_ax = plt.subplot2grid((rows, cols+1), (0, 0),
                            colspan=cols_per_big_umap, rowspan=cols_per_big_umap)
    scv.pl.umap(adata, color=batch_key, ax=tmp_ax, show=False,
                frameon="artist", legend_loc="right margin")

    tmp_ax = plt.subplot2grid((rows, cols+1), (0, cols_per_big_umap),
                            colspan=cols_per_big_umap, rowspan=cols_per_big_umap)
    scv.pl.umap(adata, color=cluster_key, ax=tmp_ax, show=False,
                frameon="artist", legend_loc="right margin")


    #################################################### 小图UMAP部分####################################################
    # cmap = plt.get_cmap("viridis") # 统一配色
    cmap = plt.get_cmap("gnuplot")  # 统一配色
    size = 120000/adata.shape[0]  # 统一细胞大小
    # 设置colorbar
    neighbor_key = "knn"
    # vmax=adata.obs[neighbor_key].max()
    # vmin, vmax = adata.obs[neighbor_key].quantile([0.25,0.75]) # 上下四分位数
    vmin, vmax = adata.obs[neighbor_key].quantile([0.05, 0.95])  # 上下95分位数
    tmp_ax = ax[cols_per_big_umap][0]
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    tmp_ax.set_aspect(1.0/(tmp_ax.get_data_ratio()),
                    adjustable="box")  # 设置子图到下边界的距离
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        cax=tmp_ax, orientation="horizontal", fraction=0.5)
    cbar.set_label("%s\nmax=%s" % (neighbor_key, int(
        adata.obs[neighbor_key].max())), fontsize=30)
    for i in range(n_batch_list):
        batch = batch_list[i]
        adata_batch = adata[adata.obs[batch_key] == batch]
        row = int(i/cols)
        col = i % cols+1
        if col == 1 and row != 0:
            ax[row+cols_per_big_umap][0].axis("off")  # 清空边框刻度
        row += cols_per_big_umap
        tmp_ax = ax[row][col]
        scv.pl.umap(adata_batch,
                    color=neighbor_key,
                    show=False,
                    title="%s/%s/%s" % (neighbor_key, batch, adata_batch.shape[0]),
                    size=size,
                    vmin=0,
                    vmax=vmax,
                    colorbar=False,
                    frameon="artist",
                    cmap=cmap,
                    ax=tmp_ax)
        # 最后一行后面几列清空边框刻度
        if i == n_batch_list-1 and (not (col == cols-1)):
            for j in range(col+1, cols+1):
                ax[row][j].axis("off") 

    neighbor_key = "bnn"
    # vmax=adata.obs[neighbor_key].max()
    # vmin, vmax = adata.obs[neighbor_key].quantile([0.25,0.75]) # 上下四分位数
    vmin, vmax = adata.obs[neighbor_key].quantile([0.05, 0.95])  # 上下95分位数
    tmp_ax = ax[rows_per_group+cols_per_big_umap][0]
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    tmp_ax.set_aspect(1.0/(tmp_ax.get_data_ratio()),
                    adjustable="box")  # 设置子图到下边界的距离
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        cax=tmp_ax, orientation="horizontal", fraction=0.5)
    cbar.set_label("%s\nmax=%s" % (neighbor_key, int(
        adata.obs[neighbor_key].max())), fontsize=30)
    for i in range(n_batch_list):
        batch = batch_list[i]
        adata_batch = adata[adata.obs[batch_key] == batch]
        row = int(i/cols)
        col = i % cols+1
        if col == 1 and row != 0:
            ax[row+rows_per_group+cols_per_big_umap][0].axis("off")  # 清空边框刻度
        row += rows_per_group+cols_per_big_umap
        tmp_ax = ax[row][col]
        scv.pl.umap(adata_batch,
                    color=neighbor_key,
                    show=False,
                    title="%s/%s/%s" % (neighbor_key, batch, adata_batch.shape[0]),
                    size=size,
                    vmin=0,
                    vmax=vmax,
                    colorbar=False,
                    frameon="artist",
                    cmap=cmap,
                    ax=tmp_ax)
        # 最后一行后面几列清空边框刻度
        if i == n_batch_list-1 and (not (col == cols-1)):
            for j in range(col+1, cols+1):
                ax[row][j].axis("off")
    
    if not save==None:
        fig.savefig(save, format = save.split(".")[-1] if "." in save else "png")
        plt.close(fig)
    return fig