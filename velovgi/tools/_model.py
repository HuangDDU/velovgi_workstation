import logging
import warnings
from functools import partial
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from anndata import AnnData
from joblib import Parallel, delayed
from scipy.stats import ttest_ind
from scvi._compat import Literal
from scvi._utils import _doc_params
from scvi.data import AnnDataManager
from scvi.data.fields import LayerField, NumericalObsField
from scvi.dataloaders import AnnDataLoader, DataSplitter
from scvi.model._utils import scrna_raw_counts_properties
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin, VAEMixin
from scvi.model.base._utils import _de_core
from scvi.train import TrainingPlan, TrainRunner
from scvi.utils._docstrings import doc_differential_expression, setup_anndata_dsp
from sklearn.metrics.pairwise import cosine_similarity

from velovi._model import VELOVI, _softplus_inverse # 用于继承的大流程模型，继承VELOVI实现VELOVGI

from velovi._constants import REGISTRY_KEYS  # 注册U，S的常量
from ._module import VELOVGAE  # VAE变为VGAE，作为主模型本体

from .dataloader._neighbor_data_splitter import NeighborDataSplitter # 使用了Neighbor MiniBatch策略的DataSplitter
from .dataloader._random_data_splitter import RandomDataSplitter # 随机样本划分
from .dataloader._cluster_data_splitter import ClusterDataSplitter # 聚类划分
from .dataloader._random_ann_dataloader import RandomAnnDataLoader

logger = logging.getLogger(__name__)



class VELOVGI(VELOVI):
    """VELOVGI

    VELOVGI
    
    Parameters
    ----------
    adata
        _description_
    n_hidden 
        _description_. Defaults to 256.
    n_latent
        _description_. Defaults to 10.
    n_layers
        _description_. Defaults to 1.
    dropout_rate
        _description_. Defaults to 0.1.
    train_batch_mode
        _description_. Defaults to "random-batch".
    gamma_init_data
        _description_. Defaults to False.
    linear_decoder
        _description_. Defaults to False.
    """
    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 256,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        train_batch_mode: Literal["random-batch", "allsample-batch"] = "random-batch",
        gamma_init_data: bool = False,
        linear_decoder: bool = False,
        **model_kwargs,
    ):
        """__init__

        Args:
            adata (AnnData): _description_
            n_hidden (int, optional): _description_. Defaults to 256.
            n_latent (int, optional): _description_. Defaults to 10.
            n_layers (int, optional): _description_. Defaults to 1.
            dropout_rate (float, optional): _description_. Defaults to 0.1.
            train_batch_mode (Literal[&quot;random, optional): _description_. Defaults to "random-batch".
            gamma_init_data (bool, optional): _description_. Defaults to False.
            linear_decoder (bool, optional): _description_. Defaults to False.
        """
        super().__init__(adata)

        self.train_batch_mode = train_batch_mode  # 批次训练模式

        self.n_latent = n_latent

        spliced = self.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)
        unspliced = self.adata_manager.get_from_registry(REGISTRY_KEYS.U_KEY)

        sorted_unspliced = np.argsort(unspliced, axis=0)
        ind = int(adata.n_obs * 0.99)
        us_upper_ind = sorted_unspliced[ind:, :]

        us_upper = []
        ms_upper = []
        for i in range(len(us_upper_ind)):
            row = us_upper_ind[i]
            us_upper += [unspliced[row, np.arange(adata.n_vars)][np.newaxis, :]]
            ms_upper += [spliced[row, np.arange(adata.n_vars)][np.newaxis, :]]
        us_upper = np.median(np.concatenate(us_upper, axis=0), axis=0)
        ms_upper = np.median(np.concatenate(ms_upper, axis=0), axis=0)

        alpha_unconstr = _softplus_inverse(us_upper)
        alpha_unconstr = np.asarray(alpha_unconstr).ravel()

        alpha_1_unconstr = np.zeros(us_upper.shape).ravel()
        lambda_alpha_unconstr = np.zeros(us_upper.shape).ravel()

        if gamma_init_data:
            gamma_unconstr = np.clip(_softplus_inverse(us_upper / ms_upper), None, 10)
        else:
            gamma_unconstr = None

        self.module = VELOVGAE(
            n_input=self.summary_stats["n_vars"],
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            gamma_unconstr_init=gamma_unconstr,
            alpha_unconstr_init=alpha_unconstr,
            alpha_1_unconstr_init=alpha_1_unconstr,
            lambda_alpha_unconstr_init=lambda_alpha_unconstr,
            switch_spliced=ms_upper,
            switch_unspliced=us_upper,
            linear_decoder=linear_decoder,
            **model_kwargs,
        )
        self._model_summary_string = (
            "VELOVGI Model with the following params: \nn_hidden: {}, n_latent: {}, n_layers: {}, dropout_rate: "
            "{}"
        ).format(
            n_hidden,
            n_latent,
            n_layers,
            dropout_rate,
        )
        # 此处的layer与NeighborMiniBatch的num_neighbors相关，保存下来
        self.n_layers = n_layers
        self.init_params_ = self._get_init_params(locals())

    # TODO: 之前是整图训练，现在开始使用Neighbor MiniBatch策略训练
    def train(
        self,
        batch_mode: Literal["neighbor", "cluster", "random", "all"] = "neighbor",
        num_neighbors: List = [3, ],  # 与Neighbor MiniBatch策略相关，同时与神经网络层数相关
        max_epochs: Optional[int] = 500,
        lr: float = 1e-2,
        weight_decay: float = 1e-2,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        batch_size: int = 256,
        early_stopping: bool = True,
        gradient_clip_val: float = 10,
        plan_kwargs: Optional[dict] = None,
        **trainer_kwargs,
    ):
        """Train the VeloVGAE Model

        Args:
            batch_mode (Literal[&quot;neighbor&quot;, &quot;cluster&quot;, &quot;random&quot;, &quot;all&quot;], optional): _description_. Defaults to "neighbor".
            num_neighbors (List, optional): _description_. Defaults to [3, ].
            lr (float, optional): _description_. Defaults to 1e-2.
            weight_decay (float, optional): _description_. Defaults to 1e-2.
            use_gpu (Optional[Union[str, int, bool]], optional): _description_. Defaults to None.
            train_size (float, optional): _description_. Defaults to 0.9.
            validation_size (Optional[float], optional): _description_. Defaults to None.
            batch_size (int, optional): _description_. Defaults to 256.
            early_stopping (bool, optional): _description_. Defaults to True.
            gradient_clip_val (float, optional): _description_. Defaults to 10.
            plan_kwargs (Optional[dict], optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        user_plan_kwargs = (
            plan_kwargs.copy() if isinstance(plan_kwargs, dict) else dict()
        )
        plan_kwargs = dict(lr=lr, weight_decay=weight_decay, optimizer="AdamW")
        plan_kwargs.update(user_plan_kwargs)

        user_train_kwargs = trainer_kwargs.copy()
        trainer_kwargs = dict(gradient_clip_val=gradient_clip_val)
        trainer_kwargs.update(user_train_kwargs)

        if batch_mode == "neighbor":
            # TODO: 直接修改为Neighbor采样的Mini-Batch策略
            # print("选择 Neighbor 策略")
            print("choosing neighbor minibatch")
            assert self.n_layers == len(num_neighbors),\
                f"Neighbor MiniBatch parameter 'num_neighbors':{num_neighbors} not math netowrk layers num:{self.n_layers}!"
            data_splitter = NeighborDataSplitter(
                self.adata_manager,
                train_size=train_size,
                validation_size=validation_size,
                num_neighbors=num_neighbors,
                use_gpu=use_gpu,
                batch_size=batch_size
            )
        elif batch_mode == "cluster":
            # TODO: 按照样本聚类划分Mini-Batch策略
            # print("选择 Cluster 策略")
            print("choosing cluster minibatch")
            data_splitter = ClusterDataSplitter(
                self.adata_manager,
                train_size=train_size,
                validation_size=validation_size,
                use_gpu=use_gpu
            )
        else:
            if batch_mode == "all":
                batch_size = self.adata.shape[0]
                # print("选择 All 策略", batch_size)
                print("choosing all minibatch")
            else:
                # print("选择 Random 策略", batch_size)
                print("choosing random minibatch")
            # TODO: 随机样本划分
            data_splitter = RandomDataSplitter(
                self.adata_manager,
                train_size=train_size,
                validation_size=validation_size,
                use_gpu=use_gpu,
                batch_size=batch_size
            )

        training_plan = TrainingPlan(self.module, **plan_kwargs)

        es = "early_stopping"
        trainer_kwargs[es] = (
            early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]
        )
        # 这里添加日志输出
        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            **trainer_kwargs,
        )
        return runner()

    def _make_data_loader(
        self,
        adata: AnnData,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = None,
    ):
        # TODO: 此处构造随机图的MiniBatch，因为最后只进行一次正向传播，最好还是整图输入
        if batch_size == None:
            # 默认整图一次输入
            batch_size = adata.shape[0]
        adata_manager = self.get_anndata_manager(adata)
        return RandomAnnDataLoader(
            adata_manager,
            indices=range(adata.shape[0]),
            device=self.device,
            shuffle=False,
            batch_size=batch_size
        )
