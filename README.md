# VeloVGI

## Introduction

![pipeline](/img/pipeline.png)
Graph Variational Autoencoder for scRNA-seq velocity

## Installation
You are suggested to reproduce the conda environment with `velovgi.yml` using the following command :
```bash
conda env create -f velovgi.yml 
```

You can get the packages also by pip command:
```
pip install velovgi
```

The main packages version are as following :
```
# Optimal Transportation
pot = 0.9.0

# Deeplearning 
torch==1.12.1
pytorch-lightning==1.7.7
torch-geometric==2.0.1

# scRNA-seq 
scanpy==1.9.3
scvi-tools==0.19.0

# RNA velocity 
scvelo==0.2.5
velovi==0.2.0
```

## Quick start

We provide a [jupyter notebook](./notebook/erythroid_lineage.ipynb) for users to quickly understand the use of the tool and the output results.

## Reference

The paper of the work is published by [link](https://bmcbiol.biomedcentral.com/articles/10.1186/s12915-024-02085-8). You can cite our work with the following methods.

```
@article{huang2024accurate,
  title={Accurate RNA velocity estimation based on multibatch network reveals complex lineage in batch scRNA-seq data},
  author={Huang, Zhaoyang and Guo, Xinyang and Qin, Jie and Gao, Lin and Ju, Fen and Zhao, Chenguang and Yu, Liang},
  journal={BMC biology},
  volume={22},
  number={1},
  pages={290},
  year={2024},
  publisher={Springer}
}
```
