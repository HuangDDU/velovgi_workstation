#############################################################################################
# AnnDataLoader部分
import copy
import logging
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from scvi.data import AnnDataManager
from scvi.dataloaders._anntorchdataset import AnnTorchDataset

logger = logging.getLogger(__name__)

#############################################################################################
# NeighborLoader部分
from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.loader.base import DataLoaderIterator
from torch_geometric.loader.utils import filter_data

#############################################################################################
# 导入的继承类
from torch_geometric.loader.neighbor_loader import NeighborSampler
# 其他
import scipy.sparse as sp  # 稀疏矩阵的边提取
from torch.utils.data._utils.collate import default_convert  # 自动转化为Tensor


class NeighborAnnDataLoader(DataLoader):
    def __init__(
        self,
        adata_manager: AnnDataManager,
        indices=None,
        num_neighbors=[3, 3],
        device="cuda:0",
        **data_loader_kwargs  # DataLoader的参数可调整, batch_size, shuffle等
    ):
        #############################################################################################
        # AnnDataLoader部分，这里会自动获取注册的所有键
        self.ann_torch_dataset = AnnTorchDataset(adata_manager)
        #############################################################################################
        # NeighborLoader部分
        self.adata = adata_manager.adata
        self.filter_per_worker = False

        # 直接使用序号做样本特征
        if indices is None:
            indices = range(self.adata.shape[0])  # 没有提供样本序号的话，使用所有样本序号作为特征
        self.input_nodes = range(len(indices))  # 样本在indices中的下标，作为Sampler和NeighborSampler的输入
        x = torch.Tensor(indices)  # 样本在adata的下标，作为PyG的Data特征
        # 利用稀疏矩阵提取边
        adj = self.adata.obsp["connectivities"][indices, :][:, indices]  # 只提取当前有效元素的邻接矩阵
        graph_coo = sp.coo_matrix(adj)
        edge_index = torch.LongTensor(np.array([graph_coo.row, graph_coo.col])).to(device)  # 边对
        edge_weight = torch.FloatTensor(graph_coo.data).to(device)  # 边权
        self.data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)  # 构造PyG的Data

        # 构造NeighborSampler
        self.neighbor_sampler = NeighborSampler(self.data, num_neighbors=num_neighbors)  # 游走邻居的Sampler
        #############################################################################################
        super().__init__(self.input_nodes, collate_fn=self.collate_fn, **data_loader_kwargs)

    def collate_fn(self, index: Union[List[int], Tensor]) -> Any:
        out = self.neighbor_sampler(index)
        if self.filter_per_worker:
            # We execute `filter_fn` in the worker process.
            out = self.filter_fn(out)
        return out

    def filter_fn(self, out: Any) -> Dict:
        # 这里直接简化，在同质图上
        node, row, col, edge, batch_size = out
        data = filter_data(self.data, node, row, col, edge,
                           self.neighbor_sampler.perm)
        data.batch_size = batch_size

        # return data if self.transform is None else self.transform(data)
        # 直接转化为字典，除了PyG的adata的信息外，还有ann_torch_dataset中提取注册的键值对
        # print(f"=====稍后转化为字典：{data}=====")
        # PyG的adata的信息转化为字典
        output_dict = {
            "id": data.x.int(),
            "edge_index": data.edge_index,
            "edge_weight": data.edge_weight,
            "batch_size": data.batch_size,
        }
        # 从ann_torch_dataset中提取注册的键值对
        adata_dict = self.ann_torch_dataset.__getitem__(data.x.int().detach().cpu().numpy())
        adata_dict = default_convert(adata_dict)  # 字典中的键转化为Tensor
        # 合并
        for k, v in adata_dict.items():
            output_dict[k] = v
        return output_dict

    def _get_iterator(self) -> Iterator:
        if self.filter_per_worker:
            return super()._get_iterator()
        # We execute `filter_fn` in the main process.
        return DataLoaderIterator(super()._get_iterator(), self.filter_fn)
