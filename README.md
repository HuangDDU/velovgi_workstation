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

The paper of the work is under review. You can cite our work with the following methods.
```
```
