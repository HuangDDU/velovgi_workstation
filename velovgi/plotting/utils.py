import math
import numpy as np
import matplotlib.pyplot as plt

# 固定一列元素个数计算画布
def calc_fig(n, cols=5 ,item_figsize=(5, 4)):
    rows = math.ceil(n/cols)
    if rows == 1:
        cols = n
    figsize = (item_figsize[0]*cols, item_figsize[1]*rows) # 整体figsize
    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    if type(ax) == np.ndarray:
        ax = ax.flatten()
    else:
        ax = np.array(ax)
    return fig, ax