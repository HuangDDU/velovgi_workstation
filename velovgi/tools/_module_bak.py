from typing import Callable, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
# from scvi._compat import Literal
from typing import Literal
from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data
from scvi.nn import Encoder, FCLayers
from torch import nn as nn
from torch.distributions import Categorical, Dirichlet, MixtureSameFamily, Normal
from torch.distributions import kl_divergence as kl

from velovi._module import VELOVAE, DecoderVELOVI
from velovi._constants import REGISTRY_KEYS  # 注册U，S的常量

from .nn._gnn_component import GNN_Encoder
from scvi.module.base import LossOutput  # 使用LossOutput代替LossRecorder

class VELOVGAE(VELOVAE):
    def __init__(
        self,
        n_input: int,
        nn_type: Literal["FC", "GCN", "GIN", "GAT"] = "GCN",  # 此处选择图卷积层的方式
        true_time_switch: Optional[np.ndarray] = None,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        log_variational: bool = False,
        latent_distribution: str = "normal",
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_observed_lib_size: bool = True,
        var_activation: Optional[Callable] = torch.nn.Softplus(),
        model_steady_states: bool = True,
        gamma_unconstr_init: Optional[np.ndarray] = None,
        alpha_unconstr_init: Optional[np.ndarray] = None,
        alpha_1_unconstr_init: Optional[np.ndarray] = None,
        lambda_alpha_unconstr_init: Optional[np.ndarray] = None,
        switch_spliced: Optional[np.ndarray] = None,
        switch_unspliced: Optional[np.ndarray] = None,
        t_max: float = 20,
        penalty_scale: float = 0.2,
        dirichlet_concentration: float = 0.25,
        linear_decoder: bool = False,
        time_dep_transcription_rate: bool = False,
    ):
        # TODO: 只修改其中的编码器部分，添加图结构的GNNEncoder
        super().__init__(
            n_input,
            true_time_switch,
            n_hidden,
            n_latent,
            n_layers,
            dropout_rate,
            log_variational,
            latent_distribution,
            use_batch_norm,
            use_layer_norm,
            use_observed_lib_size,
            var_activation,
            model_steady_states,
            gamma_unconstr_init,
            alpha_unconstr_init,
            alpha_1_unconstr_init,
            lambda_alpha_unconstr_init,
            switch_spliced,
            switch_unspliced,
            t_max,
            penalty_scale,
            dirichlet_concentration,
            linear_decoder,
            time_dep_transcription_rate,
        )
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.latent_distribution = latent_distribution
        self.use_observed_lib_size = use_observed_lib_size
        self.n_input = n_input
        self.model_steady_states = model_steady_states
        self.t_max = t_max
        self.penalty_scale = penalty_scale
        self.dirichlet_concentration = dirichlet_concentration
        self.time_dep_transcription_rate = time_dep_transcription_rate

        if switch_spliced is not None:
            self.register_buffer("switch_spliced", torch.from_numpy(switch_spliced))
        else:
            self.switch_spliced = None
        if switch_unspliced is not None:
            self.register_buffer("switch_unspliced", torch.from_numpy(switch_unspliced))
        else:
            self.switch_unspliced = None

        n_genes = n_input * 2

        # switching time
        self.switch_time_unconstr = torch.nn.Parameter(7 + 0.5 * torch.randn(n_input))
        if true_time_switch is not None:
            self.register_buffer("true_time_switch", torch.from_numpy(true_time_switch))
        else:
            self.true_time_switch = None

        # degradation
        if gamma_unconstr_init is None:
            self.gamma_mean_unconstr = torch.nn.Parameter(-1 * torch.ones(n_input))
        else:
            self.gamma_mean_unconstr = torch.nn.Parameter(
                torch.from_numpy(gamma_unconstr_init)
            )

        # splicing
        # first samples around 1
        self.beta_mean_unconstr = torch.nn.Parameter(0.5 * torch.ones(n_input))

        # transcription
        if alpha_unconstr_init is None:
            self.alpha_unconstr = torch.nn.Parameter(0 * torch.ones(n_input))
        else:
            self.alpha_unconstr = torch.nn.Parameter(
                torch.from_numpy(alpha_unconstr_init)
            )

        # TODO: Add `require_grad`
        if alpha_1_unconstr_init is None:
            self.alpha_1_unconstr = torch.nn.Parameter(0 * torch.ones(n_input))
        else:
            self.alpha_1_unconstr = torch.nn.Parameter(
                torch.from_numpy(alpha_1_unconstr_init)
            )
        self.alpha_1_unconstr.requires_grad = time_dep_transcription_rate

        if lambda_alpha_unconstr_init is None:
            self.lambda_alpha_unconstr = torch.nn.Parameter(0 * torch.ones(n_input))
        else:
            self.lambda_alpha_unconstr = torch.nn.Parameter(
                torch.from_numpy(lambda_alpha_unconstr_init)
            )
        self.lambda_alpha_unconstr.requires_grad = time_dep_transcription_rate

        # likelihood dispersion
        # for now, with normal dist, this is just the variance
        self.scale_unconstr = torch.nn.Parameter(-1 * torch.ones(n_genes, 4))

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"
        self.use_batch_norm_decoder = use_batch_norm_decoder

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        n_input_encoder = n_genes
        # TODO: 此处需要把编码器调整为图结构
        self.z_encoder = GNN_Encoder(
            n_input_encoder,
            n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            nn_type=nn_type,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            activation_fn=torch.nn.ReLU,
        )
        # decoder goes from n_latent-dimensional space to n_input-d data
        n_input_decoder = n_latent
        self.decoder = DecoderVELOVI(
            n_input_decoder,
            n_input,
            n_layers=n_layers,
            n_hidden=n_hidden,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
            activation_fn=torch.nn.ReLU,
            linear_decoder=linear_decoder,
        )

    def _get_inference_input(self, tensors):

        input_dict = {}
        # TODO: 这里直接从Batch结果里拿边，边权，样本个数
        input_dict["edge_index"] = tensors["edge_index"]
        input_dict["edge_weight"] = tensors["edge_weight"]
        input_dict["batch_size"] = tensors["batch_size"]

        spliced = tensors[REGISTRY_KEYS.X_KEY]
        unspliced = tensors[REGISTRY_KEYS.U_KEY]
        # 这里最后可能需要转换类型，最后获得latent representation时会出错
        device = tensors["edge_index"].device.type
        if not(spliced.device.type == device):
            # print("Transfer spliced, unspliced :", device)
            spliced = spliced.to(device)
            unspliced = unspliced.to(device)

        input_dict["spliced"] = spliced
        input_dict["unspliced"] = unspliced
        # 简单看一下Neighbor策略下输入样本个数，与真实batch_size的关系
        # print("input sample num", spliced.shape[0])
        # print("batch_size", tensors["batch_size"])

        return input_dict

    def inference(
        self,
        spliced,
        unspliced,
        edge_index,
        edge_weight,
        batch_size,
        n_samples=1,
    ):
        spliced_ = spliced
        unspliced_ = unspliced
        if self.log_variational:
            spliced_ = torch.log(0.01 + spliced)
            unspliced_ = torch.log(0.01 + unspliced)

        encoder_input = torch.cat((spliced_, unspliced_), dim=-1)

        # TODO: 此处正向传播添加图结构，同时这里因为NeighborLoader的策略要选择前batch_size个样本
        qz_m, qz_v, z = self.z_encoder(encoder_input, edge_index, edge_weight)
        qz_m, qz_v, z = qz_m[:batch_size], qz_v[:batch_size], z[:batch_size]  # 前batch_size个样本

        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            # when z is normal, untran_z == z
            untran_z = Normal(qz_m, qz_v.sqrt()).sample()
            z = self.z_encoder.z_transformation(untran_z)

        # 获取降解、剪切、初始转录、终止转录、转录变化因子
        gamma, beta, alpha, alpha_1, lambda_alpha = self._get_rates()

        outputs = dict(
            z=z,
            qz_m=qz_m,
            qz_v=qz_v,
            gamma=gamma,
            beta=beta,
            alpha=alpha,
            alpha_1=alpha_1,
            lambda_alpha=lambda_alpha,
        )
        return outputs

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
        n_obs: float = 1.0,
    ):
        # TODO: 计算损失的前batch_size个样本的X,U特征
        batch_size = tensors["batch_size"]
        batch_sample_num = tensors[REGISTRY_KEYS.X_KEY].shape[0]
        tensors[REGISTRY_KEYS.X_KEY] = tensors[REGISTRY_KEYS.X_KEY][:batch_size]
        tensors[REGISTRY_KEYS.U_KEY] = tensors[REGISTRY_KEYS.U_KEY][:batch_size]
        # return super().loss(
        #     tensors,
        #     inference_outputs,
        #     generative_outputs,
        #     kl_weight,
        #     n_obs,
        # )
        # 复制之前velovi的代码
        spliced = tensors[REGISTRY_KEYS.X_KEY]
        unspliced = tensors[REGISTRY_KEYS.U_KEY]

        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]

        px_pi = generative_outputs["px_pi"]
        px_pi_alpha = generative_outputs["px_pi_alpha"]

        end_penalty = generative_outputs["end_penalty"]
        mixture_dist_s = generative_outputs["mixture_dist_s"]
        mixture_dist_u = generative_outputs["mixture_dist_u"]

        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(0, 1)).sum(dim=1)

        reconst_loss_s = -mixture_dist_s.log_prob(spliced)
        reconst_loss_u = -mixture_dist_u.log_prob(unspliced)

        reconst_loss = reconst_loss_u.sum(dim=-1) + reconst_loss_s.sum(dim=-1)

        kl_pi = kl(
            Dirichlet(px_pi_alpha),
            Dirichlet(self.dirichlet_concentration * torch.ones_like(px_pi)),
        ).sum(dim=-1)

        # local loss
        kl_local = kl_divergence_z + kl_pi
        weighted_kl_local = kl_weight * (kl_divergence_z) + kl_pi

        local_loss = torch.mean(reconst_loss + weighted_kl_local)

        # combine local and global
        global_loss = 0
        loss = (
            local_loss
            + self.penalty_scale * (1 - kl_weight) * end_penalty
            + (1 / n_obs) * kl_weight * (global_loss)
        )

        # loss_recoder = LossRecorder(
        #     loss, reconst_loss, kl_local, torch.tensor(global_loss)
        # )
        loss_ouput = LossOutput(
            loss=loss,
            reconstruction_loss=reconst_loss,
            kl_local=kl_local,
            kl_global=torch.tensor(global_loss),
            extra_metrics={
                "batch_size": batch_size,
                "batch_sample_num" : batch_sample_num
            }
        )

        return loss_ouput
