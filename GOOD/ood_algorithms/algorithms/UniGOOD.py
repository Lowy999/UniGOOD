import torch
from torch import Tensor
from torch_geometric.data import Batch

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from GOOD.utils.initial import reset_random_seed
from GOOD.utils.train import at_stage
from .BaseOOD import BaseOODAlg


@register.ood_alg_register
class UniGOOD(BaseOODAlg):
    r"""
    Implementation of the UniGOOD algorithm

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(UniGOOD, self).__init__(config)
        self.mean_loss = None
        self.spec_loss = None

        self.causal_z1_sy = None
        self.causal_z2_sy = None
        self.causal_uz1_sy = None
        self.causal_uz2_sy = None
        self.target = None

        self.conf_z1_sy = None
        self.conf_z2_sy = None

        self.causal_z1_ce = None
        self.causal_z2_ce = None

        self.conf_uz1_ce = None
        self.conf_uz2_ce = None

        self.mean = None
        self.logstd = None

        alpha = config.ood.extra_param[0]
        beta = config.ood.extra_param[1]
        self.c1 = 2 * alpha
        self.c2 = 2 * beta
        self.c3 = alpha**2
        self.c4 = 2 * alpha * beta
        self.c5 = beta**2
        self.kl_rate = config.ood.extra_param[2]
        self.conf_sy_rate = config.ood.extra_param[3]
        self.causal_ce_rate = config.ood.extra_param[4]
        self.conf_ce_rate = config.ood.extra_param[5]

    def stage_control(self, config: Union[CommonArgs, Munch]):
        r"""
        Set valuables before each epoch. Largely used for controlling multi-stage training and epoch related parameter
        settings.

        Args:
            config: munchified dictionary of args.

        """
        if self.stage == 0 and at_stage(1, config):
            reset_random_seed(config)
            self.stage = 1
        self.kl_rate = (
            min(config.train.epoch / config.train.max_epoch, 1.0)
            * config.ood.extra_param[2]
        )

    def process(self, labeled_data, wild_data):
        if wild_data == None:
            return self.model(data=labeled_data)
        else:
            return self.model(labeled_data=labeled_data, wild_data=wild_data)

    def output_postprocess(self, model_output: Tensor, **kwargs) -> Tensor:
        r"""
        Process the raw output of model

        Args:
            model_output (Tensor): model raw output

        Returns (Tensor):
            model raw predictions.

        """
        (
            self.causal_z1_sy,
            self.causal_z2_sy,
            self.causal_uz1_sy,
            self.causal_uz2_sy,
            self.target,
            self.conf_z1_sy,
            self.conf_z2_sy,
            self.causal_z1_ce,
            self.causal_z2_ce,
            self.conf_uz1_ce,
            self.conf_uz2_ce,
            self.mean,
            self.logstd,
        ) = model_output
        return self.causal_z1_sy

    def loss_calculate(
        self,
        raw_pred: Tensor,
        targets: Tensor,
        mask: Tensor,
        config: Union[CommonArgs, Munch],
    ) -> Tensor:
        r"""
        Calculate loss based on UniGOOD algorithm

        Args:
            raw_pred (Tensor): model predictions
            targets (Tensor): input labels
            mask (Tensor): NAN masks for data formats
            config (Union[CommonArgs, Munch]): munchified dictionary of args


        Returns (Tensor):
            loss based on UniGOOD algorithm

        """
        self.mean_loss, self.spec_loss = self.sup_D(
            self.causal_z1_sy,
            self.causal_z2_sy,
            self.causal_uz1_sy,
            self.causal_uz2_sy,
            self.target,
        )
        self.spec_loss["conf_sy_loss"] = self.conf_sy_rate * self.sup_D_loss1(
            self.conf_z1_sy, self.conf_z2_sy
        )
        self.spec_loss["causal_ce_loss"] = self.causal_ce_rate * self.sup_D_loss1(
            self.causal_z1_ce, self.causal_z2_ce
        )
        self.spec_loss["conf_ce_loss"] = self.conf_ce_rate * self.sup_D_loss2_loss5(
            self.conf_uz1_ce, self.conf_uz2_ce
        )
        self.spec_loss["kl_divergence"] = (
            -self.kl_rate
            * 0.5
            * (1 + 2 * self.logstd - self.mean**2 - torch.exp(self.logstd) ** 2).mean()
        )
        return (
            self.mean_loss
            + self.spec_loss["conf_sy_loss"]
            + self.spec_loss["causal_ce_loss"]
            + self.spec_loss["conf_ce_loss"]
            + self.spec_loss["kl_divergence"]
        )

    def loss_postprocess(
        self,
        loss: Tensor,
        labeled_data: Batch,
        wild_data: Batch,
        mask: Tensor,
        config: Union[CommonArgs, Munch],
        **kwargs
    ) -> Tensor:
        r"""
        Process loss based on UniGOOD algorithm

        Returns (Tensor):
            loss based on UniGOOD algorithm

        """
        return loss

    def set_up(self, model: torch.nn.Module, config: Union[CommonArgs, Munch], loader):
        r"""
        Training setup of optimizer and scheduler

        Args:
            model (torch.nn.Module): model for setup
            config (Union[CommonArgs, Munch]): munchified dictionary of args

        Returns:
            None

        """
        self.model: torch.nn.Module = model
        self.optimizer = torch.optim.Adam(
            self.model.subgraph_encoder.parameters(),
            lr=config.train.lr,
            weight_decay=config.train.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.train.max_epoch,
            eta_min=config.train.lr,
        )

    def backward(self, loss):
        r"""
        Gradient backward process and parameter update.

        Args:
            loss: target loss
        """
        loss.backward()
        self.optimizer.step()

    def sup_D(self, z1, z2, uz1, uz2, target, mu=1.0):
        r"""
        Spectral contrastive loss adapted from `Bridge<https://github.com/deeplearning-wisc/graph-spectral-ood>`_
        """
        device = z1.device
        bsz_l, bsz_u = len(z1), len(uz1)

        mat_ll = torch.matmul(z1, z2.T)
        mat_uu = torch.matmul(uz1, uz2.T)

        mat_lu_s2 = torch.matmul(z1, uz2.T) ** 2
        mat_ul_s2 = torch.matmul(uz1, z2.T) ** 2
        mat_ll_s2 = mat_ll**2 * (1 - torch.diag(torch.ones(bsz_l)).to(device))
        mat_uu_s2 = mat_uu**2 * (1 - torch.diag(torch.ones(bsz_u)).to(device))

        c1, c2 = self.c1, self.c2
        c3, c4, c5 = self.c3, self.c4, self.c5

        target_ = target.contiguous().view(-1, 1)
        pos_labeled_mask = torch.eq(target_, target_.T).to(device)
        cls_sample_count = pos_labeled_mask.sum(1)

        loss1 = -c1 * torch.sum((mat_ll * pos_labeled_mask) / cls_sample_count**2)

        pos_unlabeled_mask = torch.diag(torch.ones(bsz_u)).to(device)
        loss2 = -c2 * torch.sum(mat_uu * pos_unlabeled_mask) / bsz_u

        loss3 = c3 * torch.sum(
            mat_ll_s2 / (cls_sample_count[:, None] * cls_sample_count[None, :])
        )

        loss4 = c4 * torch.sum(
            mat_lu_s2 / (cls_sample_count[:, None] * bsz_u)
        ) + c4 * torch.sum(mat_ul_s2 / (cls_sample_count[None, :] * bsz_u))

        loss5 = c5 * torch.sum(mat_uu_s2) / (bsz_u * (bsz_u - 1))

        return (loss1 + loss2 + loss3 + loss4 + loss5) / mu, {
            "loss1": loss1 / mu,
            "loss2": loss2 / mu,
            "loss3": loss3 / mu,
            "loss4": loss4 / mu,
            "loss5": loss5 / mu,
        }

    def sup_D_loss1(self, conf_z1, conf_z2, mu=1.0):
        mat_ll = torch.matmul(conf_z1, conf_z2.T)
        c1 = self.c1
        loss1 = -c1 * torch.mean(mat_ll)
        return loss1 / mu

    def sup_D_loss2_loss5(self, uz1, uz2, mu=1.0):
        device = uz1.device
        bsz_u = len(uz1)

        mat_uu = torch.matmul(uz1, uz2.T)
        mat_uu_s2 = mat_uu**2 * (1 - torch.diag(torch.ones(bsz_u)).to(device))

        c2, c5 = self.c2, self.c5

        pos_unlabeled_mask = torch.diag(torch.ones(bsz_u)).to(device)

        loss2 = -c2 * torch.sum(mat_uu * pos_unlabeled_mask) / bsz_u
        loss5 = c5 * torch.sum(mat_uu_s2) / (bsz_u * (bsz_u - 1))

        return (loss2 + loss5) / mu
