import copy
import numpy as np

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic
from .GINvirtualnode import vGINFeatExtractor
from .GINs import GINFeatExtractor


def normalized(z):
    return F.normalize(z, dim=1)


@register.model_register
class UniGOODGIN(GNNBasic):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(UniGOODGIN, self).__init__(config)
        self.subgraph_encoder = SubgraphEncoder(config.ood.ood_param, config)

        if config.dataset.dataset_type == "syn":
            self.classifier = nn.Linear(
                config.model.dim_hidden, config.dataset.num_classes
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(config.model.dim_hidden, config.model.dim_hidden),
                nn.BatchNorm1d(config.model.dim_hidden),
                nn.ReLU(),
                nn.Linear(config.model.dim_hidden, config.dataset.num_classes),
            )

        self.task = config.task

    def forward(self, *args, **kwargs):
        r"""
        The UniGOOD model implementation.

        Args:
            *args (list): argument list for the use of arguments_read.
            **kwargs (dict): key word arguments for the use of arguments_read.

        Returns (Tensor):
            Label predictions and other results for loss calculations.

        """
        if kwargs.get("wild_data") != None:

            (
                (
                    labeled_causal_rep_1_sy,
                    labeled_conf_rep_1_sy,
                    labeled_causal_rep_1_ce,
                    labeled_conf_rep_1_ce,
                ),
                (
                    labeled_causal_rep_2_sy,
                    labeled_conf_rep_2_sy,
                    labeled_causal_rep_2_ce,
                    labeled_conf_rep_2_ce,
                ),
                labeled_mean,
                labeled_logstd,
            ) = self.subgraph_encoder(
                data=kwargs.get("labeled_data"), sscl=True, distribution_id=0
            )

            (
                (
                    wild_causal_rep_1_sy,
                    wild_conf_rep_1_sy,
                    wild_causal_rep_1_ce,
                    wild_conf_rep_1_ce,
                ),
                (
                    wild_causal_rep_2_sy,
                    wild_conf_rep_2_sy,
                    wild_causal_rep_2_ce,
                    wild_conf_rep_2_ce,
                ),
                wild_mean,
                wild_logstd,
            ) = self.subgraph_encoder(
                data=kwargs.get("wild_data"), sscl=True, distribution_id=1
            )

            conf_z1_sy = normalized(
                torch.cat([labeled_conf_rep_1_sy, wild_conf_rep_1_sy], dim=0)
            )
            conf_z2_sy = normalized(
                torch.cat([labeled_conf_rep_2_sy, wild_conf_rep_2_sy], dim=0)
            )

            conf_uz1_ce = normalized(
                torch.cat([labeled_conf_rep_1_ce, wild_conf_rep_1_ce], dim=0)
            )
            conf_uz2_ce = normalized(
                torch.cat([labeled_conf_rep_2_ce, wild_conf_rep_2_ce], dim=0)
            )

            causal_z1_sy = normalized(labeled_causal_rep_1_sy)
            causal_z2_sy = normalized(labeled_causal_rep_2_sy)
            causal_uz1_sy = normalized(wild_causal_rep_1_sy)
            causal_uz2_sy = normalized(wild_causal_rep_2_sy)

            causal_z1_ce = normalized(
                torch.cat([labeled_causal_rep_1_ce, wild_causal_rep_1_ce], dim=0)
            )
            causal_z2_ce = normalized(
                torch.cat([labeled_causal_rep_2_ce, wild_causal_rep_2_ce], dim=0)
            )

            mean = torch.cat([labeled_mean, wild_mean], dim=0)
            logstd = torch.cat([labeled_logstd, wild_logstd], dim=0)

            return (
                causal_z1_sy,
                causal_z2_sy,
                causal_uz1_sy,
                causal_uz2_sy,
                kwargs.get("labeled_data").y,
                conf_z1_sy,
                conf_z2_sy,
                causal_z1_ce,
                causal_z2_ce,
                conf_uz1_ce,
                conf_uz2_ce,
                mean,
                logstd,
            )
        else:
            causal_rep = self.subgraph_encoder(
                data=kwargs.get("data"), sscl=False, distribution_id=1
            )
            return self.classifier(causal_rep), causal_rep


@register.model_register
class UniGOODvGIN(UniGOODGIN):
    r"""
    The GIN virtual node version of UniGOOD.
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(UniGOODvGIN, self).__init__(config)
        self.subgraph_encoder = SubgraphEncoder(
            config.ood.ood_param, config, virtual_node=True
        )

        replace_BatchNorm1d_with_DualBatchNorm1d(self.subgraph_encoder)


class SubgraphEncoder(nn.Module):

    def __init__(self, causal_ratio, config, **kwargs):
        super(SubgraphEncoder, self).__init__()
        config_catt = copy.deepcopy(config)
        config_catt.model.model_layer = 2
        config_fe = copy.deepcopy(config)
        config_fe.model.model_layer = config.model.model_layer - 2
        if kwargs.get("virtual_node"):
            self.gnn = vGINFeatExtractor(
                config_catt, without_readout=True, no_bn=True, **kwargs
            )
            self.gnn_std = vGINFeatExtractor(
                config_catt, without_readout=True, no_bn=True, **kwargs
            )
            self.feat_encoder_sy = vGINFeatExtractor(
                config_fe, without_embed=True, no_bn=True
            )
            self.feat_encoder_ce = vGINFeatExtractor(
                config_fe, without_embed=True, no_bn=True
            )
        else:
            self.gnn = GINFeatExtractor(
                config_catt, without_readout=True, no_bn=True, **kwargs
            )
            self.gnn_std = GINFeatExtractor(
                config_catt, without_readout=True, no_bn=True, **kwargs
            )
            self.feat_encoder_sy = GINFeatExtractor(
                config_fe, without_embed=True, no_bn=True
            )
            self.feat_encoder_ce = GINFeatExtractor(
                config_fe, without_embed=True, no_bn=True
            )
        self.ratio = causal_ratio

        self.dataset = config.dataset.dataset_name
        self.task = config.task

        if self.dataset not in ["GDHIVZINC"]:
            replace_batchnorm1d_with_affine(self.gnn)
            replace_batchnorm1d_with_affine(self.gnn_std)
        replace_batchnorm1d_with_affine(self.feat_encoder_sy)
        replace_batchnorm1d_with_affine(self.feat_encoder_ce)

        self.mean_encoder = nn.Sequential(
            nn.Linear(config.model.dim_hidden, 2 * config.model.dim_hidden),
            AffineNormalize(2 * config.model.dim_hidden),
            nn.ReLU(),
            nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden),
        )
        self.logstd_encoder = nn.Sequential(
            nn.Linear(config.model.dim_hidden, 2 * config.model.dim_hidden),
            AffineNormalize(2 * config.model.dim_hidden),
            nn.ReLU(),
            nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden),
        )

    def get_subgraph_rep(self, z, data, edge_score, need_conf_rep, ratio):
        if ratio < 0:
            ratio = 0
        elif ratio > 1:
            ratio = 1

        (causal_edge_index, causal_edge_attr, causal_edge_weight), (
            conf_edge_index,
            conf_edge_attr,
            conf_edge_weight,
        ) = split_graph(data, edge_score, ratio)

        causal_x, causal_edge_index, causal_batch, _ = relabel(
            z, causal_edge_index, data.batch
        )
        conf_x, conf_edge_index, conf_batch, _ = relabel(z, conf_edge_index, data.batch)

        set_masks(causal_edge_weight, self)
        causal_rep_sy = self.feat_encoder_sy(
            data=Data(
                x=causal_x,
                edge_index=causal_edge_index,
                edge_attr=causal_edge_attr,
                batch=causal_batch,
            ),
            batch_size=data.batch[-1].item() + 1,
        )
        clear_masks(self)

        if need_conf_rep == True:
            set_masks(causal_edge_weight, self)
            causal_rep_ce = self.feat_encoder_ce(
                data=Data(
                    x=causal_x,
                    edge_index=causal_edge_index,
                    edge_attr=causal_edge_attr,
                    batch=causal_batch,
                ),
                batch_size=data.batch[-1].item() + 1,
            )
            clear_masks(self)

            set_masks(conf_edge_weight, self)
            conf_rep_sy = self.feat_encoder_sy(
                data=Data(
                    x=conf_x,
                    edge_index=conf_edge_index,
                    edge_attr=conf_edge_attr,
                    batch=conf_batch,
                ),
                batch_size=data.batch[-1].item() + 1,
            )
            conf_rep_ce = self.feat_encoder_ce(
                data=Data(
                    x=conf_x,
                    edge_index=conf_edge_index,
                    edge_attr=conf_edge_attr,
                    batch=conf_batch,
                ),
                batch_size=data.batch[-1].item() + 1,
            )
            clear_masks(self)
            return causal_rep_sy, conf_rep_sy, causal_rep_ce, conf_rep_ce
        else:
            return causal_rep_sy

    def forward(self, *args, **kwargs):
        data = kwargs.get("data") or None

        for m in self.modules():
            if isinstance(m, DualBatchNorm1d):
                m.distribution_id = kwargs.get("distribution_id")

        rep = self.gnn(*args, **kwargs)
        mean = self.mean_encoder(rep)
        row, col = data.edge_index
        z = rep

        if kwargs.get("sscl") == True:
            logstd = self.logstd_encoder(self.gnn_std(*args, **kwargs))
            std = torch.exp(logstd)
            eps_1 = torch.randn_like(std)
            eps_2 = torch.randn_like(std)
            z_1 = eps_1.mul(std).add_(mean)
            z_2 = eps_2.mul(std).add_(mean)
            edge_score_1 = (z_1[row] * z_1[col]).mean(dim=1)
            edge_score_2 = (z_2[row] * z_2[col]).mean(dim=1)
            return (
                self.get_subgraph_rep(z, data, edge_score_1, True, self.ratio),
                self.get_subgraph_rep(z, data, edge_score_2, True, self.ratio),
                mean,
                logstd,
            )
        else:
            edge_score = (mean[row] * mean[col]).mean(dim=1)
            return self.get_subgraph_rep(z, data, edge_score, False, self.ratio)


def set_masks(mask: Tensor, model: nn.Module):
    r"""
    Adopted from https://github.com/wuyxin/dir-gnn.
    """
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = True
            module._explain = True
            module.__edge_mask__ = mask
            module._edge_mask = mask


def clear_masks(model: nn.Module):
    r"""
    Adopted from https://github.com/wuyxin/dir-gnn.
    """
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = False
            module._explain = False
            module.__edge_mask__ = None
            module._edge_mask = None


def split_graph(data, edge_score, ratio):
    r"""
    Adapted from https://github.com/wuyxin/dir-gnn.
    """
    has_edge_attr = (
        hasattr(data, "edge_attr") and getattr(data, "edge_attr") is not None
    )

    new_idx_reserve, new_idx_drop, _, _, _ = sparse_topk(
        edge_score, data.batch[data.edge_index[0]], ratio, descending=True
    )
    new_causal_edge_index = data.edge_index[:, new_idx_reserve]
    new_conf_edge_index = data.edge_index[:, new_idx_drop]

    new_causal_edge_weight = edge_score[new_idx_reserve]
    new_conf_edge_weight = -edge_score[new_idx_drop]

    if has_edge_attr:
        new_causal_edge_attr = data.edge_attr[new_idx_reserve]
        new_conf_edge_attr = data.edge_attr[new_idx_drop]
    else:
        new_causal_edge_attr = None
        new_conf_edge_attr = None

    return (new_causal_edge_index, new_causal_edge_attr, new_causal_edge_weight), (
        new_conf_edge_index,
        new_conf_edge_attr,
        new_conf_edge_weight,
    )


def relabel(x, edge_index, batch, pos=None):
    r"""
    Adopted from https://github.com/wuyxin/dir-gnn.
    """
    num_nodes = x.size(0)
    sub_nodes = torch.unique(edge_index)
    x = x[sub_nodes]
    batch = batch[sub_nodes]
    row, col = edge_index
    # remapping the nodes in the explanatory subgraph to new ids.
    node_idx = row.new_full((num_nodes,), -1)
    node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=row.device)
    edge_index = node_idx[edge_index]
    if pos is not None:
        pos = pos[sub_nodes]
    return x, edge_index, batch, pos


def sparse_sort(
    src: torch.Tensor, index: torch.Tensor, dim=0, descending=False, eps=1e-12
):
    r"""
    Adopted from https://github.com/rusty1s/pytorch_scatter/issues/48.
    """
    f_src = src.float()
    f_min, f_max = f_src.min(dim)[0], f_src.max(dim)[0]
    norm = (f_src - f_min) / (f_max - f_min + eps) + index.float() * (-1) ** int(
        descending
    )
    perm = norm.argsort(dim=dim, descending=descending)

    return src[perm], perm


def sparse_topk(
    src: torch.Tensor,
    index: torch.Tensor,
    ratio: float,
    dim=0,
    descending=False,
    eps=1e-12,
):
    r"""
    Sparse topk calculation.
    """
    rank, perm = sparse_sort(src, index, dim, descending, eps)
    num_nodes = degree(index, dtype=torch.long)
    k = (ratio * num_nodes.to(float)).ceil().to(torch.long)
    start_indices = torch.cat(
        [torch.zeros((1,), device=src.device, dtype=torch.long), num_nodes.cumsum(0)]
    )
    mask = [
        torch.arange(k[i], dtype=torch.long, device=src.device) + start_indices[i]
        for i in range(len(num_nodes))
    ]
    mask = torch.cat(mask, dim=0)
    mask = torch.zeros_like(index, device=index.device).index_fill(0, mask, 1).bool()
    topk_perm = perm[mask]
    exc_perm = perm[~mask]

    return topk_perm, exc_perm, rank, perm, mask


def replace_batchnorm1d_with_identity(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm1d):
            setattr(module, name, nn.Identity())
        else:
            replace_batchnorm1d_with_identity(child)


def replace_BatchNorm1d_with_DualBatchNorm1d(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm1d):
            num_features = child.num_features
            setattr(module, name, DualBatchNorm1d(num_features))
        else:
            replace_BatchNorm1d_with_DualBatchNorm1d(child)


class DualBatchNorm1d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(num_features, affine=False)
        self.bn2 = nn.BatchNorm1d(num_features, affine=False)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

        self.distribution_id = 0

    def forward(self, x):
        if self.distribution_id == 0:
            x_norm = self.bn1(x)
        else:
            x_norm = self.bn2(x)
        return self.weight * x_norm + self.bias


class AffineNormalize(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + 1e-5)
        return self.weight * x_norm + self.bias


def replace_batchnorm1d_with_affine(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm1d):
            setattr(module, name, AffineNormalize(child.num_features))
        else:
            replace_batchnorm1d_with_affine(child)


def set_batchnorm_eval(module):
    for m in module.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.eval()


def set_batchnorm_train(module):
    for m in module.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.train()
