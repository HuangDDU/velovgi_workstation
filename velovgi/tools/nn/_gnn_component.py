########################################################################################
import collections
from typing import Callable, Iterable, List, Optional

import torch
import torch.nn as nn
from torch.distributions import Normal

from torch_geometric.nn import MessagePassing, GCNConv, GATConv

from scvi.nn import one_hot
from scvi._compat import Literal

def _identity(x):
    return x


# 根据传参选择卷积块
class GNN_Layers(nn.Module):
    """
    A helper class to build fully-connected layers for a neural network.

    Parameters
    ----------
    n_in
        The dimensionality of the input
    n_out
        The dimensionality of the output
    n_cat_list
        A list containing, for each category of interest,
        the number of categories. Each category will be
        included using a one-hot encoding.
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    use_batch_norm
        Whether to have `BatchNorm` layers or not
    use_layer_norm
        Whether to have `LayerNorm` layers or not
    use_activation
        Whether to have layer activation or not
    bias
        Whether to learn bias in linear layers or not
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    activation_fn
        Which activation function to use
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        nn_type: Literal["FC", "GCN", "GIN", "GAT"] = "GCN",  # 此处选择图卷积层的方式
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        use_activation: bool = True,
        bias: bool = True,
        inject_covariates: bool = True,
        activation_fn: nn.Module = nn.ReLU,
    ):
        super().__init__()
        self.inject_covariates = inject_covariates
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []

        cat_dim = sum(self.n_cat_list)

        # TODO: 根据参数选择图卷积层
        GNNConv = None
        if nn_type == "GCN":
            GNNConv = GCNConv
        elif nn_type == "GAT":
            GNNConv = GATConv
        else:
            # 以后再加入一些其他的卷积层
            pass
        # print("GNN卷积方式:", GNNConv)
        self.gnn_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        f"Layer {i}",
                        nn.Sequential(
                            # TODO: 此处选择的卷积层即可
                            GNNConv(
                                n_in + cat_dim * self.inject_into_layer(i),
                                n_out,
                                bias=bias,
                            ),
                            # non-default params come from defaults in original Tensorflow implementation
                            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001)
                            if use_batch_norm
                            else None,
                            nn.LayerNorm(n_out, elementwise_affine=False)
                            if use_layer_norm
                            else None,
                            activation_fn() if use_activation else None,
                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(
                        zip(layers_dim[:-1], layers_dim[1:])
                    )
                ]
            )
        )


    def inject_into_layer(self, layer_num) -> bool:
        """Helper to determine if covariates should be injected."""
        user_cond = layer_num == 0 or (layer_num > 0 and self.inject_covariates)
        return user_cond

    def set_online_update_hooks(self, hook_first_layer=True):
        """Set online update hooks."""
        self.hooks = []

        def _hook_fn_weight(grad):
            categorical_dims = sum(self.n_cat_list)
            new_grad = torch.zeros_like(grad)
            if categorical_dims > 0:
                new_grad[:, -categorical_dims:] = grad[:, -categorical_dims:]
            return new_grad

        def _hook_fn_zero_out(grad):
            return grad * 0

        for i, layers in enumerate(self.gnn_layers):
            for layer in layers:
                if i == 0 and not hook_first_layer:
                    continue
                if isinstance(layer, nn.Linear):
                    if self.inject_into_layer(i):
                        w = layer.weight.register_hook(_hook_fn_weight)
                    else:
                        w = layer.weight.register_hook(_hook_fn_zero_out)
                    self.hooks.append(w)
                    b = layer.bias.register_hook(_hook_fn_zero_out)
                    self.hooks.append(b)

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.LongTensor,
                edge_weight: torch.FloatTensor,
                *cat_list: int):
        """
        Forward computation on ``x``.

        Parameters
        ----------
        x
            tensor of values with shape ``(n_in,)``
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        :class:`torch.Tensor`
            tensor of shape ``(n_out,)``
        """
        one_hot_cat_list = []  # for generality in this list many indices useless.

        if len(self.n_cat_list) > len(cat_list):
            raise ValueError(
                "nb. categorical args provided doesn't match init. params."
            )
        for n_cat, cat in zip(self.n_cat_list, cat_list):
            if n_cat and cat is None:
                raise ValueError("cat not provided while n_cat != 0 in init. params.")
            if n_cat > 1:  # n_cat = 1 will be ignored - no additional information
                if cat.size(1) != n_cat:
                    one_hot_cat = one_hot(cat, n_cat)
                else:
                    one_hot_cat = cat  # cat has already been one_hot encoded
                one_hot_cat_list += [one_hot_cat]
        for i, layers in enumerate(self.gnn_layers):
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, nn.BatchNorm1d):
                        if x.dim() == 3:
                            x = torch.cat(
                                [(layer(slice_x)).unsqueeze(0) for slice_x in x], dim=0
                            )
                        else:
                            x = layer(x)
                    else:
                        if isinstance(layer, MessagePassing) and self.inject_into_layer(i):
                            if x.dim() == 3:
                                one_hot_cat_list_layer = [
                                    o.unsqueeze(0).expand(
                                        (x.size(0), o.size(0), o.size(1))
                                    )
                                    for o in one_hot_cat_list
                                ]
                            else:
                                one_hot_cat_list_layer = one_hot_cat_list
                            x = torch.cat((x, *one_hot_cat_list_layer), dim=-1)
                            # TODO: 此处NN加了图结构
                            # if not (x.device.type == "cuda" and edge_index.device.type == "cuda" and edge_weight.device.type == "cuda"):
                            #     # 后面提取latent representation时可能会出错，都转化为cpu
                            #     print("图结构转化为cpu")
                            #     x = x.cpu()
                            #     edge_index = edge_index.cpu()
                            #     edge_weight = edge_weight.cpu()
                            # else:
                            #     # print("图结构保持为gpu")
                            #     pass
                            x = layer(x, edge_index, edge_weight)
                        else:
                            x = layer(x)  # 其他层直接输出即可
        return x

# 根据传参选择卷积块
class GNN_Encoder(nn.Module):
    """
    Encode data of ``n_input`` dimensions into a latent space of ``n_output`` dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (data space)
    n_output
        The dimensionality of the output (latent space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    distribution
        Distribution of z
    var_eps
        Minimum value for the variance;
        used for numerical stability
    var_activation
        Callable used to ensure positivity of the variance.
        Defaults to :meth:`torch.exp`.
    return_dist
        Return directly the distribution of z instead of its parameters.
    **kwargs
        Keyword args for :class:`~scvi.nn.FCLayers`
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        nn_type: Literal["FC", "GCN", "GIN", "GAT"] = "GCN",  # 此处选择图卷积层的方式
        dropout_rate: float = 0.1,
        distribution: str = "normal",
        var_eps: float = 1e-4,
        var_activation: Optional[Callable] = None,
        return_dist: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.distribution = distribution
        self.var_eps = var_eps

        #
        self.encoder = GNN_Layers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            nn_type=nn_type,
            dropout_rate=dropout_rate,
            **kwargs,
        )
        # 均值和方差也是需要重新选择
        # TODO: 根据参数选择图卷积层
        GNNConv = None
        if nn_type == "GCN":
            GNNConv = GCNConv
        elif nn_type == "GAT":
            GNNConv = GATConv
        else:
            # 以后再加入一些其他的卷积层
            pass
        # print("GNN_Encoder上GNN卷积方式:", GNNConv)
        self.mean_encoder = GNNConv(n_hidden, n_output)
        self.var_encoder = GNNConv(n_hidden, n_output)
        self.return_dist = return_dist

        if distribution == "ln":
            self.z_transformation = nn.Softmax(dim=-1)
        else:
            self.z_transformation = _identity
        self.var_activation = torch.exp if var_activation is None else var_activation

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.LongTensor,
                edge_weight: torch.FloatTensor,
                *cat_list: int):
        r"""
        The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\)
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim Ne(q_m, \\mathbf{I}q_v) \\)

        Parameters
        ----------
        x
            tensor with shape (n_input,)
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        3-tuple of :py:class:`torch.Tensor`
            tensors of shape ``(n_latent,)`` for mean and var, and sample

        """
        # Parameters for latent distribution
        # TODO: 此处添加了图结构的输入
        q = self.encoder(x, edge_index, edge_weight, *cat_list)
        q_m = self.mean_encoder(q, edge_index, edge_weight)
        q_v = self.var_activation(self.var_encoder(q, edge_index, edge_weight)) + self.var_eps
        dist = Normal(q_m, q_v.sqrt())
        latent = self.z_transformation(dist.rsample())
        if self.return_dist:
            return dist, latent
        return q_m, q_v, latent

if __name__=="__main__":
    print("====================开始测试GVAE====================")
    # 模型参数
    n_sample = 5
    n_input_encoder = 30  # VAE的编码器输入维度
    n_latent = 10  # VAE的隐变量维度，对应编码器的输出维度
    n_layers = 2  # VAE共享层数
    n_hidden = 16  # VAE编码器的隐层维度

    # 特征矩阵与图（手动构造）
    X = torch.rand((n_sample, n_input_encoder))
    edge_index = torch.LongTensor([[0, 0, 0, 0],
                                  [1, 2, 3, 4]])
    edge_weight = torch.FloatTensor([0.25, 0.25, 0.25, 0.25])

    # 编码器
    z_encode_gcn = GNN_Encoder(
        n_input_encoder,
        n_latent,
        n_layers=n_layers,
        n_hidden=n_hidden,
        nn_type="GCN"
    )
    print("GCN构造完成", z_encode_gcn)
    z_encode_gat = GNN_Encoder(
        n_input_encoder,
        n_latent,
        n_layers=n_layers,
        n_hidden=n_hidden,
        nn_type="GAT"
    )
    print("GAT构造完成", z_encode_gat)

    # 执行正向传播
    output = z_encode_gcn(X, edge_index, edge_weight)
    print("GCN输出", output)
    output = z_encode_gcn(X, edge_index, edge_weight)
    print("GAT输出", output)