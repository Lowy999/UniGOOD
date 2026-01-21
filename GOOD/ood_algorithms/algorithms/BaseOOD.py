"""
Base class for OOD algorithms
"""
import torch
import faiss
from abc import ABC
import numpy as np
from torch import Tensor
from torch_geometric.data import Batch
import sklearn.metrics as sk

from GOOD.utils.config_reader import Union, CommonArgs, Munch
from typing import Tuple
from GOOD.utils.initial import reset_random_seed
from GOOD.utils.train import at_stage
from GOOD.utils.train import nan2zero_get_mask


class BaseOODAlg(ABC):
    r"""
    Base class for OOD algorithms

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(BaseOODAlg, self).__init__()
        self.optimizer: torch.optim.Adam = None
        self.scheduler: torch.optim.lr_scheduler._LRScheduler = None
        self.model: torch.nn.Module = None

        self.mean_loss = None
        self.spec_loss = None
        self.stage = 0

    def stage_control(self, config):
        r"""
        Set valuables before each epoch. Largely used for controlling multi-stage training and epoch related parameter
        settings.

        Args:
            config: munchified dictionary of args.

        """
        if self.stage == 0 and at_stage(1, config):
            reset_random_seed(config)
            self.stage = 1

    def preprocess(self, loader):
        return
    
    def process(self, labeled_data, wild_data):
        return self.model(data=labeled_data)

    def output_postprocess(self, model_output: Tensor, **kwargs) -> Tensor:
        r"""
        Process the raw output of model

        Args:
            model_output (Tensor): model raw output

        Returns (Tensor):
            model raw predictions

        """
        pred, fea = model_output
        return pred

    def loss_calculate(self, raw_pred: Tensor, targets: Tensor, mask: Tensor, 
                       config: Union[CommonArgs, Munch]) -> Tensor:
        r"""
        Calculate loss

        Args:
            raw_pred (Tensor): model predictions
            targets (Tensor): input labels
            mask (Tensor): NAN masks for data formats
            config (Union[CommonArgs, Munch]): munchified dictionary of args


        Returns (Tensor):
            cross entropy loss

        """
        loss = config.metric.loss_func(raw_pred, targets, reduction='none') * mask
        return loss

    def loss_postprocess(self, loss: Tensor, labeled_data: Batch, wild_data: Batch, mask: Tensor, config: Union[CommonArgs, Munch],
                         **kwargs) -> Tensor:
        r"""
        Process loss

        Returns (Tensor):
            processed loss

        """
        self.mean_loss = loss.sum() / mask.sum()
        return self.mean_loss

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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.train.lr,
                                          weight_decay=config.train.weight_decay)
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

    def step_epoch(self, loader, config):
        self.scheduler.step()

    def detection_evaluate(self, loader, config):
        r"""
        OOD detection evaluation adapted from `Bridge<https://github.com/deeplearning-wisc/graph-spectral-ood>`_
        """

        stat = {"fpr": None, "auroc": None}
        self.model.eval()
        normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)

        f_train_labeled_in = []
        for idx, train_labeled_in_data in enumerate(loader["train_labeled_in"]):
            current_bs = train_labeled_in_data.batch[-1] + 1
                    
            if idx == 100:
                break
            train_labeled_in_data = train_labeled_in_data.to(config.device)
            mask, _ = nan2zero_get_mask(train_labeled_in_data, config)
            if mask is None:
                continue
            with torch.set_grad_enabled(False):
                _, fea = self.process(train_labeled_in_data, None)
            f_train_labeled_in.append(fea[:current_bs].cpu().numpy())
        f_train_labeled_in = normalizer(np.concatenate(f_train_labeled_in))

        f_test_in = []
        for idx, test_in_data in enumerate(loader["test_in"]):
            current_bs = test_in_data.batch[-1] + 1

            test_in_data = test_in_data.to(config.device)
            mask, _ = nan2zero_get_mask(test_in_data, config)
            if mask is None:
                continue
            with torch.set_grad_enabled(False):
                _, fea = self.process(test_in_data, None)
            f_test_in.append(fea[:current_bs].cpu().numpy())
        f_test_in = normalizer(np.concatenate(f_test_in))

        f_test_out = []
        for idx, test_out_data in enumerate(loader["test_out"]):
            current_bs = test_out_data.batch[-1] + 1

            test_out_data = test_out_data.to(config.device)
            mask, _ = nan2zero_get_mask(test_out_data, config)
            if mask is None:
                continue
            with torch.set_grad_enabled(False):
                _, fea = self.process(test_out_data, None)
            f_test_out.append(fea[:current_bs].cpu().numpy())
        f_test_out = normalizer(np.concatenate(f_test_out))

        rand_ind = np.random.choice(f_train_labeled_in.shape[0], f_train_labeled_in.shape[0], replace=False)
        index = faiss.IndexFlatL2(f_train_labeled_in.shape[1])
        index.add(f_train_labeled_in[rand_ind])

        knn_k = [50]

        knn_in_dis = np.zeros((f_test_in.shape[0], len(knn_k)))
        knn_out_dis = np.zeros((f_test_out.shape[0], len(knn_k)))

        for bid, k in enumerate(knn_k):        
            D_test, _ = index.search(f_test_in, k)
            scores_test = -D_test[:, -1]

            D_ood_test, _ = index.search(f_test_out, k)
            scores_ood_test = -D_ood_test[:, -1]

            knn_in_dis[:, bid] = scores_test
            knn_out_dis[:, bid] = scores_ood_test
            stat["auroc"], stat["fpr"] = get_measures(scores_test, scores_ood_test)

            print("#IN#\nTest_out KNN K={} ==> Test FPR={:.2f}%, AUROC={:.2f}%".format(
                    k, stat["fpr"]*100, stat["auroc"]*100))

        self.model.train()

        return {"fpr": stat["fpr"], "auroc": stat["auroc"]}
    
def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out

def fpr_and_fdr_at_recall_4linear(y_true, y_score, recall_level=0.95, thres=0.0, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or np.array_equal(classes, [-1, 1]) or
                 np.array_equal(classes, [0]) or np.array_equal(classes, [-1]) or
                 np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps # add one because of zero-based indexing
 
    thresholds = y_score[threshold_idxs]
    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]

    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]
    cutoff = np.argmin(np.abs(recall - recall_level))
    
    return tps[cutoff]/np.sum(y_true), fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])

def get_measures(_pos, _neg, recall_level=0.95):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1      # pos label = 1

    auroc = sk.roc_auc_score(labels, examples)
    _, fpr = fpr_and_fdr_at_recall_4linear(labels, examples, recall_level)
    return auroc, fpr

