import matplotlib.pyplot as plt
import seaborn as sns

def box_plot(df, x, y, hue, ax, orient="v", palette ="Set1"):

    g = sns.boxplot(
        data=df, 
        y=y, x=x, hue=hue,ax=ax,
        orient=orient, palette = palette, fliersize=0,
        showmeans=True, notch=False, # 不用奇形怪状了
        meanprops={"marker":"o",
                   "markerfacecolor":"white", 
                   "markeredgecolor":"black",
                   "markersize":"8"}
    )
    return g


def plot_metric_total_df(df):
    if "_" in list(df["Model"])[0]:
        # 区分模型和数据
        print("区分模型和数据")
        df["hue"] = df["Model"].apply(lambda x:x.split("_")[0])
        df["row"] = df["Model"].apply(lambda x:x.split("_")[1])
    else:
        # 在一组数据上的模型
        print("在一组数据上的模型")
        df["hue"] = df["Model"]
        df["row"] = "Metric"

    palette = "Set1" # 配色风格
    orient = "v"
    fig,ax=plt.subplots(2,1,sharex=False, figsize=(5,10))
    ax=ax.flatten()

    box_plot(df.loc[df["Metric"].isin(["CBDir"])], x="row", 
                y="CBDir", orient=orient, hue="hue",ax=ax[0], palette = palette)
    ax[0].legend(loc="lower left", bbox_to_anchor=(1, 0.25), fontsize=14)

    box_plot(df.loc[df["Metric"].isin(["ICVCoh"])], x="row", 
                y="ICVCoh", orient=orient, hue="hue",ax=ax[1], palette = palette)
    ax[1].legend(loc="lower left", bbox_to_anchor=(1, 0.25), fontsize=14)