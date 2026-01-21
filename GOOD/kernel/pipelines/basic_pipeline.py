r"""Training pipeline: training/evaluation structure, batch training."""

import datetime
import os
import shutil
import time
import copy
from typing import Dict
from typing import Union
import sklearn.metrics as sk

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import torch
import torch.nn
from munch import Munch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm

from GOOD.ood_algorithms.algorithms.BaseOOD import BaseOODAlg
from GOOD.utils.args import CommonArgs
from GOOD.utils.evaluation import eval_data_preprocess, eval_score
from GOOD.utils.register import register
from GOOD.utils.train import nan2zero_get_mask


@register.pipeline_register
class Pipeline:
    r"""
    Kernel pipeline.

    Args:
        task (str): Current running task.
        model (torch.nn.Module): The GNN model.
        loader (Union[DataLoader, Dict[str, DataLoader]]): The data loader.
        ood_algorithm (BaseOODAlg): The OOD algorithm.
        config (Union[CommonArgs, Munch]): Please refer to :ref:`configs:GOOD Configs and command line Arguments (CA)`.

    """

    def __init__(
        self,
        task: str,
        model: torch.nn.Module,
        loader: Union[DataLoader, Dict[str, DataLoader]],
        ood_algorithm: BaseOODAlg,
        config: Union[CommonArgs, Munch],
    ):
        super(Pipeline, self).__init__()
        self.task: str = task
        self.model: torch.nn.Module = model
        self.loader: Union[DataLoader, Dict[str, DataLoader]] = loader
        self.ood_algorithm: BaseOODAlg = ood_algorithm
        self.config: Union[CommonArgs, Munch] = config
        self.rng = np.random.default_rng(config.random_seed)

        self.early_stop_counter = 0

    def mix_batches(self, wild_in_data, wild_co_data, wild_out_data):
        pi_co = self.config.dataset.pi_co
        pi_out = self.config.dataset.pi_out

        total_samples = len(wild_in_data.y)
        n_co = int(total_samples * pi_co)
        n_out = int(total_samples * pi_out)
        n_in = total_samples - n_co - n_out

        idx_in = self.rng.permutation(len(wild_in_data.y))[:n_in]
        idx_co = self.rng.permutation(len(wild_co_data.y))[:n_co]
        idx_out = self.rng.permutation(len(wild_out_data.y))[:n_out]

        all_data = (
            [wild_in_data[i] for i in idx_in]
            + [wild_co_data[i] for i in idx_co]
            + [wild_out_data[i] for i in idx_out]
        )

        self.rng.shuffle(all_data)

        merged_batch = Batch.from_data_list(all_data, exclude_keys=["env_id"])
        return merged_batch

    def train_batch(self, labeled_data, wild_data=None) -> dict:
        r"""
        Train a batch. (Project use only)

        Args:
            data (Batch): Current batch of data.

        Returns:
            Calculated loss.
        """
        if wild_data == None:
            if self.config.model.model_name in ["UniGOODGIN", "UniGOODvGIN"]:
                self.ood_algorithm.model.subgraph_encoder.eval()

        labeled_data = labeled_data.to(self.config.device)
        if wild_data != None:
            wild_data = wild_data.to(self.config.device)

        self.ood_algorithm.optimizer.zero_grad()

        mask, targets = nan2zero_get_mask(labeled_data, self.config)
        model_output = self.ood_algorithm.process(labeled_data, wild_data)
        raw_pred = self.ood_algorithm.output_postprocess(model_output)
        loss = self.ood_algorithm.loss_calculate(raw_pred, targets, mask, self.config)
        loss = self.ood_algorithm.loss_postprocess(
            loss, labeled_data, wild_data, mask, self.config
        )
        self.ood_algorithm.backward(loss)

    def pretrain(self):
        r"""
        Pretraining pipeline. (Project use only)
        """

        # config model
        print("#D#Config model")
        self.config_model("pretrain")

        # Load training utils
        print("#D#Load training utils")
        self.ood_algorithm.set_up(self.model, self.config, self.loader)

        # train the model
        for epoch in range(self.config.train.max_epoch):
            self.config.train.epoch = epoch
            print(f"#IN#Epoch {epoch}:")

            mean_loss = 0
            spec_loss = 0

            self.ood_algorithm.stage_control(self.config)

            start = time.time()
            if self.config.ood.wild_data == True:
                self.loader["train_labeled_in"].dataset.offset = self.rng.integers(
                    len(self.loader["train_labeled_in"].dataset)
                )
                self.loader["train_wild_in"].dataset.offset = self.rng.integers(
                    len(self.loader["train_wild_in"].dataset)
                )
                self.loader["train_wild_co"].dataset.offset = self.rng.integers(
                    len(self.loader["train_wild_co"].dataset)
                )
                self.loader["train_wild_out"].dataset.offset = self.rng.integers(
                    len(self.loader["train_wild_out"].dataset)
                )
                train_loaders = enumerate(
                    zip(
                        self.loader["train_labeled_in"],
                        self.loader["train_wild_in"],
                        self.loader["train_wild_co"],
                        self.loader["train_wild_out"],
                    )
                )

                for index, (
                    labeled_data,
                    wild_in_data,
                    wild_co_data,
                    wild_out_data,
                ) in train_loaders:
                    if labeled_data.batch is not None and (
                        labeled_data.batch[-1] < self.config.train.train_bs / 2 - 1
                    ):
                        continue

                    wild_data = self.mix_batches(
                        wild_in_data, wild_co_data, wild_out_data
                    )

                    # train a batch
                    self.train_batch(labeled_data=labeled_data, wild_data=wild_data)
                    mean_loss = (mean_loss * index + self.ood_algorithm.mean_loss) / (
                        index + 1
                    )

                    if self.ood_algorithm.spec_loss is not None:
                        if isinstance(self.ood_algorithm.spec_loss, dict):
                            desc = f"ML: {mean_loss:.4f}|"
                            for (
                                loss_name,
                                loss_value,
                            ) in self.ood_algorithm.spec_loss.items():
                                if not isinstance(spec_loss, dict):
                                    spec_loss = dict()
                                if loss_name not in spec_loss.keys():
                                    spec_loss[loss_name] = 0
                                spec_loss[loss_name] = (
                                    spec_loss[loss_name] * index + loss_value
                                ) / (index + 1)
                                desc += f"{loss_name}: {spec_loss[loss_name]:.4f}|"
                        else:
                            spec_loss = (
                                spec_loss * index + self.ood_algorithm.spec_loss
                            ) / (index + 1)

                end = time.time()
                print(f"#IN#Pretraining Time taken: {end - start:.2f} seconds")

                # Epoch val
                print("#IN#\nPretraining...")
                if self.ood_algorithm.spec_loss is not None:
                    if isinstance(self.ood_algorithm.spec_loss, dict):
                        desc = f"ML: {mean_loss:.4f}|"
                        for (
                            loss_name,
                            loss_value,
                        ) in self.ood_algorithm.spec_loss.items():
                            desc += f"{loss_name}: {spec_loss[loss_name]:.4f}|"
                        print(f"#IN#Approximated " + desc[:-1])
                    else:
                        print(
                            f"#IN#Approximated average M/S Loss {mean_loss:.4f}/{spec_loss:.4f}"
                        )
                else:
                    print(
                        f"#IN#Approximated average training loss {mean_loss.cpu().item():.4f}"
                    )

                start = time.time()
                val_stat = self.wild_evaluate()
                end = time.time()
                print(f"#IN#Inference Time taken: {end - start:.2f} seconds")
                test_out_stat = self.ood_algorithm.detection_evaluate(
                    self.loader, self.config
                )

                # checkpoints save
                self.unsupervised_save_epoch(
                    epoch,
                    val_stat,
                    test_out_stat,
                    self.config,
                )

            else:
                for index, data in enumerate(self.loader["train_labeled_in"]):
                    if data.batch is not None and (
                        data.batch[-1] < self.config.train.train_bs - 1
                    ):
                        continue

                    # train a batch
                    self.train_batch(data)
                    mean_loss = (mean_loss * index + self.ood_algorithm.mean_loss) / (
                        index + 1
                    )

                    if self.ood_algorithm.spec_loss is not None:
                        if isinstance(self.ood_algorithm.spec_loss, dict):
                            desc = f"ML: {mean_loss:.4f}|"
                            for (
                                loss_name,
                                loss_value,
                            ) in self.ood_algorithm.spec_loss.items():
                                if not isinstance(spec_loss, dict):
                                    spec_loss = dict()
                                if loss_name not in spec_loss.keys():
                                    spec_loss[loss_name] = 0
                                spec_loss[loss_name] = (
                                    spec_loss[loss_name] * index + loss_value
                                ) / (index + 1)
                                desc += f"{loss_name}: {spec_loss[loss_name]:.4f}|"
                        else:
                            spec_loss = (
                                spec_loss * index + self.ood_algorithm.spec_loss
                            ) / (index + 1)

                end = time.time()
                print(f"#IN#Pretraining Time taken: {end - start:.2f} seconds")

                # Epoch val
                print("#IN#\nPretraining...")
                if self.ood_algorithm.spec_loss is not None:
                    if isinstance(self.ood_algorithm.spec_loss, dict):
                        desc = f"ML: {mean_loss:.4f}|"
                        for (
                            loss_name,
                            loss_value,
                        ) in self.ood_algorithm.spec_loss.items():
                            desc += f"{loss_name}: {spec_loss[loss_name]:.4f}|"
                        print(f"#IN#Approximated " + desc[:-1])
                    else:
                        print(
                            f"#IN#Approximated average M/S Loss {mean_loss:.4f}/{spec_loss:.4f}"
                        )
                else:
                    print(
                        f"#IN#Approximated average training loss {mean_loss.cpu().item():.4f}"
                    )

                start = time.time()
                val_in_stat = self.evaluate("val_labeled_in")
                end = time.time()
                print(f"#IN#Inference Time taken: {end - start:.2f} seconds")
                test_in_stat = self.evaluate("test_in")
                test_co_stat = self.evaluate("test_co")
                test_out_stat = {"fpr": None, "auroc": None}

                # checkpoints save
                self.save_epoch(
                    epoch,
                    val_in_stat,
                    test_in_stat,
                    test_co_stat,
                    test_out_stat,
                    self.config,
                )

            # --- scheduler step ---
            self.ood_algorithm.step_epoch(self.loader, self.config)

        print("#IN#Training end.")

    def train(self):
        r"""
        Training pipeline. (Project use only)
        """
        # config model
        print("#D#Config model")
        self.config_model("train")

        # Load training utils
        print("#D#Load training utils")
        self.ood_algorithm.set_up(self.model, self.config, self.loader)

        # train the model
        for epoch in range(self.config.train.max_epoch):
            self.config.train.epoch = epoch
            print(f"#IN#Epoch {epoch}:")

            mean_loss = 0
            spec_loss = 0

            self.ood_algorithm.stage_control(self.config)

            start = time.time()
            if self.config.ood.wild_data == True:
                self.loader["train_labeled_in"].dataset.offset = self.rng.integers(
                    len(self.loader["train_labeled_in"].dataset)
                )
                self.loader["train_wild_in"].dataset.offset = self.rng.integers(
                    len(self.loader["train_wild_in"].dataset)
                )
                self.loader["train_wild_co"].dataset.offset = self.rng.integers(
                    len(self.loader["train_wild_co"].dataset)
                )
                self.loader["train_wild_out"].dataset.offset = self.rng.integers(
                    len(self.loader["train_wild_out"].dataset)
                )
                train_loaders = enumerate(
                    zip(
                        self.loader["train_labeled_in"],
                        self.loader["train_wild_in"],
                        self.loader["train_wild_co"],
                        self.loader["train_wild_out"],
                    )
                )
                for index, (
                    labeled_data,
                    wild_in_data,
                    wild_co_data,
                    wild_out_data,
                ) in train_loaders:
                    if labeled_data.batch is not None and (
                        labeled_data.batch[-1] < self.config.train.train_bs / 2 - 1
                    ):
                        continue

                    wild_data = self.mix_batches(
                        wild_in_data, wild_co_data, wild_out_data
                    )

                    self.ood_algorithm.preprocess(train_loaders)

                    # train a batch
                    self.train_batch(labeled_data=labeled_data, wild_data=wild_data)
                    mean_loss = (mean_loss * index + self.ood_algorithm.mean_loss) / (
                        index + 1
                    )

                    if self.ood_algorithm.spec_loss is not None:
                        if isinstance(self.ood_algorithm.spec_loss, dict):
                            desc = f"ML: {mean_loss:.4f}|"
                            for (
                                loss_name,
                                loss_value,
                            ) in self.ood_algorithm.spec_loss.items():
                                if not isinstance(spec_loss, dict):
                                    spec_loss = dict()
                                if loss_name not in spec_loss.keys():
                                    spec_loss[loss_name] = 0
                                spec_loss[loss_name] = (
                                    spec_loss[loss_name] * index + loss_value
                                ) / (index + 1)
                                desc += f"{loss_name}: {spec_loss[loss_name]:.4f}|"
                        else:
                            spec_loss = (
                                spec_loss * index + self.ood_algorithm.spec_loss
                            ) / (index + 1)
            else:
                for index, data in enumerate(self.loader["train_labeled_in"]):
                    if data.batch is not None and (
                        data.batch[-1] < self.config.train.train_bs - 1
                    ):
                        continue

                    # Parameter for DANN
                    p = (
                        index / len(self.loader["train_labeled_in"]) + epoch
                    ) / self.config.train.max_epoch
                    self.config.train.alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1

                    self.ood_algorithm.preprocess(self.loader["train_labeled_in"])

                    # train a batch
                    self.train_batch(data)
                    mean_loss = (mean_loss * index + self.ood_algorithm.mean_loss) / (
                        index + 1
                    )

                    if self.ood_algorithm.spec_loss is not None:
                        if isinstance(self.ood_algorithm.spec_loss, dict):
                            desc = f"ML: {mean_loss:.4f}|"
                            for (
                                loss_name,
                                loss_value,
                            ) in self.ood_algorithm.spec_loss.items():
                                if not isinstance(spec_loss, dict):
                                    spec_loss = dict()
                                if loss_name not in spec_loss.keys():
                                    spec_loss[loss_name] = 0
                                spec_loss[loss_name] = (
                                    spec_loss[loss_name] * index + loss_value
                                ) / (index + 1)
                                desc += f"{loss_name}: {spec_loss[loss_name]:.4f}|"
                        else:
                            spec_loss = (
                                spec_loss * index + self.ood_algorithm.spec_loss
                            ) / (index + 1)

            end = time.time()
            print(f"#IN#Training Time taken: {end - start:.2f} seconds")

            # Epoch val
            print("#IN#\nTraining...")
            if self.ood_algorithm.spec_loss is not None:
                if isinstance(self.ood_algorithm.spec_loss, dict):
                    desc = f"ML: {mean_loss:.4f}|"
                    for loss_name, loss_value in self.ood_algorithm.spec_loss.items():
                        desc += f"{loss_name}: {spec_loss[loss_name]:.4f}|"
                    print(f"#IN#Approximated " + desc[:-1])
                else:
                    print(
                        f"#IN#Approximated average M/S Loss {mean_loss:.4f}/{spec_loss:.4f}"
                    )
            else:
                print(
                    f"#IN#Approximated average training loss {mean_loss.cpu().item():.4f}"
                )

            start = time.time()
            val_in_stat = self.evaluate("val_labeled_in")
            end = time.time()
            print(f"#IN#Inference Time taken: {end - start:.2f} seconds")
            test_in_stat = self.evaluate("test_in")
            test_co_stat = self.evaluate("test_co")
            test_out_stat = self.ood_algorithm.detection_evaluate(
                self.loader, self.config
            )

            # checkpoints save
            self.save_epoch(
                epoch,
                val_in_stat,
                test_in_stat,
                test_co_stat,
                test_out_stat,
                self.config,
            )

            # --- scheduler step ---
            self.ood_algorithm.step_epoch(self.loader, self.config)

        print("#IN#Training end.")

    def evaluate(self, split: str):
        r"""
        This function is design to collect data results and calculate scores and loss given a dataset subset.
        (For project use only)

        Returns:
            A score and a loss.

        """
        stat = {"score": None, "loss": None}
        if self.loader.get(split) is None:
            return stat
        self.model.eval()

        loss_all = []
        mask_all = []
        pred_all = []
        target_all = []
        pbar = self.loader[split]
        for data in pbar:
            data: Batch = data.to(self.config.device)

            mask, targets = nan2zero_get_mask(data, self.config)
            if mask is None:
                return stat
            with torch.set_grad_enabled(False):
                model_output = self.ood_algorithm.process(data, None)
            raw_preds = self.ood_algorithm.output_postprocess(model_output)

            # --------------- Loss collection ------------------
            loss: torch.tensor = (
                self.config.metric.loss_func(raw_preds, targets, reduction="none")
                * mask
            )
            mask_all.append(mask)
            loss_all.append(loss)

            # ------------- Score data collection ------------------
            pred, target = eval_data_preprocess(data.y, raw_preds, mask, self.config)
            pred_all.append(pred)
            target_all.append(target)

        # ------- Loss calculate -------
        loss_all = torch.cat(loss_all)
        mask_all = torch.cat(mask_all)
        stat["loss"] = loss_all.sum() / mask_all.sum()

        # --------------- Metric calculation including ROC_AUC, Accuracy, AP.  --------------------
        stat["score"] = eval_score(pred_all, target_all, self.config)

        print(
            f'#IN#\n{split.capitalize()} {self.config.metric.score_name}: {stat["score"]:.4f}\n'
            f'{split.capitalize()} Loss: {stat["loss"]:.4f}'
        )

        self.model.train()

        return {"score": stat["score"], "loss": stat["loss"]}

    def wild_evaluate(self):
        self.model.eval()

        loss_all = []
        mean_loss = 0
        spec_loss = 0

        val_loaders = enumerate(
            zip(
                self.loader["val_labeled_in"],
                self.loader["val_wild_in"],
                self.loader["val_wild_co"],
                self.loader["val_wild_out"],
            )
        )
        for index, (
            labeled_data,
            wild_in_data,
            wild_co_data,
            wild_out_data,
        ) in val_loaders:
            wild_data = self.mix_batches(wild_in_data, wild_co_data, wild_out_data)

            labeled_data = labeled_data.to(self.config.device)
            wild_data = wild_data.to(self.config.device)

            mask, targets = nan2zero_get_mask(labeled_data, self.config)
            with torch.set_grad_enabled(False):
                model_output = self.ood_algorithm.process(labeled_data, wild_data)

            raw_pred = self.ood_algorithm.output_postprocess(model_output)

            # --------------- Loss collection ------------------
            loss = self.ood_algorithm.loss_calculate(
                raw_pred, targets, mask, self.config
            )
            loss = self.ood_algorithm.loss_postprocess(
                loss, labeled_data, wild_data, mask, self.config
            )
            loss_all.append(loss)

            mean_loss = (mean_loss * index + self.ood_algorithm.mean_loss) / (index + 1)

            if self.ood_algorithm.spec_loss is not None:
                if isinstance(self.ood_algorithm.spec_loss, dict):
                    desc = f"ML: {mean_loss:.4f}|"
                    for (
                        loss_name,
                        loss_value,
                    ) in self.ood_algorithm.spec_loss.items():
                        if not isinstance(spec_loss, dict):
                            spec_loss = dict()
                        if loss_name not in spec_loss.keys():
                            spec_loss[loss_name] = 0
                        spec_loss[loss_name] = (
                            spec_loss[loss_name] * index + loss_value
                        ) / (index + 1)
                        desc += f"{loss_name}: {spec_loss[loss_name]:.4f}|"
                else:
                    spec_loss = (spec_loss * index + self.ood_algorithm.spec_loss) / (
                        index + 1
                    )

        print("#IN#\nEvaluating...")
        if self.ood_algorithm.spec_loss is not None:
            if isinstance(self.ood_algorithm.spec_loss, dict):
                desc = f"ML: {mean_loss:.4f}|"
                for loss_name, loss_value in self.ood_algorithm.spec_loss.items():
                    desc += f"{loss_name}: {spec_loss[loss_name]:.4f}|"
                print(f"#IN#Approximated " + desc[:-1])
            else:
                print(
                    f"#IN#Approximated average M/S Loss {mean_loss:.4f}/{spec_loss:.4f}"
                )
        else:
            print(
                f"#IN#Approximated average training loss {mean_loss.cpu().item():.4f}"
            )

        self.model.train()

        return {"loss": sum(loss_all) / len(loss_all)}

    def load_task(self):
        r"""
        Launch a training or a test.
        """
        if self.task == "pretrain":
            self.pretrain()
        elif self.task == "train":
            self.train()
        elif self.task == "test":
            # config model
            print("#D#Config model and output the best checkpoint info...")
            self.config_model(self.task)

    def config_model(self, mode: str, load_param=False):
        r"""
        A model configuration utility. Responsible for transiting model from CPU -> GPU and loading checkpoints.
        Args:
            mode (str): 'train' or 'test'.
            load_param: When True, loading test checkpoint will load parameters to the GNN model.

        Returns:
            Test score and loss if mode=='test'.
        """
        self.model.to(self.config.device)
        self.model.train()

        # load checkpoint
        if mode == "train":
            if self.config.model.model_name in ["UniGOODGIN", "UniGOODvGIN"]:
                ckpt_path = os.path.join(
                    self.config.ckpt_dir.replace("ERM_", "UniGOOD_"),
                    "pretrain",
                    f"best.ckpt",
                )
                print(f"#IN#Loading {ckpt_path}")
                ckpt = torch.load(ckpt_path)
                self.model.load_state_dict(ckpt["state_dict"])
                for param in self.model.subgraph_encoder.parameters():
                    param.requires_grad = False

        if mode == "test":
            try:
                ckpt = torch.load(
                    self.config.test_ckpt, map_location=self.config.device
                )
            except FileNotFoundError:
                print(
                    f"#E#Checkpoint not found at {os.path.abspath(self.config.test_ckpt)}"
                )
                exit(1)
            print(f'#IN#Loading best Checkpoint {ckpt["epoch"]}...')
            print(
                f'#IN#Checkpoint {ckpt["epoch"]}: \n-----------------------------------\n'
                f'In {self.config.metric.score_name}: {ckpt["test_in_stat_score"] * 100:.2f}\n'
                f'OOD {self.config.metric.score_name}: {ckpt["test_co_stat_score"] * 100:.2f}\n'
                f'FPR: {ckpt["test_out_stat_fpr"] * 100:.2f}\n'
                f'AUROC: {ckpt["test_out_stat_auroc"] * 100:.2f}\n'
                f'{ckpt["test_in_stat_score"] * 100:.2f},{ckpt["test_co_stat_score"] * 100:.2f},{ckpt["test_out_stat_fpr"] * 100:.2f},{ckpt["test_out_stat_auroc"] * 100:.2f}\n'
            )

            if load_param:
                if self.config.ood.ood_alg != "EERM":
                    self.model.load_state_dict(ckpt["state_dict"])
                else:
                    self.model.gnn.load_state_dict(ckpt["state_dict"])

    def save_epoch(
        self,
        epoch: int,
        val_in_stat: dir,
        test_in_stat: dir,
        test_co_stat: dir,
        test_out_stat: dir,
        config: Union[CommonArgs, Munch],
    ):
        r"""
        Training util for checkpoint saving.

        Returns:
            None

        """
        state_dict = (
            self.model.state_dict()
            if config.ood.ood_alg != "EERM"
            else self.model.gnn.state_dict()
        )
        ckpt = {
            "state_dict": state_dict,
            "val_in_stat_score": val_in_stat["score"],
            "val_in_stat_loss": val_in_stat["loss"],
            "test_in_stat_score": test_in_stat["score"],
            "test_in_stat_loss": test_in_stat["loss"],
            "test_co_stat_score": test_co_stat["score"],
            "test_co_stat_loss": test_co_stat["loss"],
            "test_out_stat_fpr": test_out_stat["fpr"],
            "test_out_stat_auroc": test_out_stat["auroc"],
            "time": datetime.datetime.now().strftime("%b%d %Hh %M:%S"),
            "model": {
                "model name": f"{config.model.model_name} {config.model.model_level} layers",
                "dim_hidden": config.model.dim_hidden,
                "dim_ffn": config.model.dim_ffn,
                "global pooling": config.model.global_pool,
            },
            "dataset": config.dataset.dataset_name,
            "train": {
                "weight_decay": config.train.weight_decay,
                "learning_rate": config.train.lr,
                "mile stone": config.train.mile_stones,
                "Batch size": f"{config.train.train_bs}, {config.train.val_bs}, {config.train.test_bs}",
            },
            "OOD": {
                "OOD alg": config.ood.ood_alg,
                "OOD param": config.ood.ood_param,
                "number of environments": config.dataset.num_envs,
            },
            "log file": config.log_path,
            "epoch": epoch,
            "max epoch": config.train.max_epoch,
        }
        if epoch < config.train.pre_train:
            return

        if not os.path.exists(os.path.join(config.ckpt_dir, config.task)):
            os.makedirs(os.path.join(config.ckpt_dir, config.task))
            print(
                f"#W#Directory does not exists. Have built it automatically.\n"
                f"{os.path.abspath(os.path.join(config.ckpt_dir, 'train'))}"
            )

        saved_file = None
        if (
            config.metric.best_stat["score"] is None
            or config.metric.lower_better * val_in_stat["score"]
            <= config.metric.lower_better * config.metric.best_stat["score"]
            or epoch % config.train.save_gap == 0
        ):
            saved_file = os.path.join(config.ckpt_dir, config.task, f"{epoch}.ckpt")
            torch.save(ckpt, saved_file)
            shutil.copy(
                saved_file, os.path.join(config.ckpt_dir, config.task, f"last.ckpt")
            )

        # --- Out-Of-Domain checkpoint ---
        if (
            config.metric.best_stat["score"] is None
            or config.metric.lower_better * val_in_stat["score"]
            <= config.metric.lower_better * config.metric.best_stat["score"]
        ):
            config.metric.best_stat["score"] = val_in_stat["score"]
            config.metric.best_stat["loss"] = val_in_stat["loss"]
            shutil.copy(
                saved_file, os.path.join(config.ckpt_dir, config.task, f"best.ckpt")
            )
            print("#IM#Saved a new best checkpoint.")
            self.early_stop_counter = 0
        else:
            self.early_stop_counter += 1
            if self.early_stop_counter >= config.train.early_stop:
                if config.task == "train":
                    self.task = "test"
                    self.load_task()
                raise ValueError(f"Early Stopping.")
        if saved_file is not None and config.clean_save:
            os.unlink(saved_file)

    def unsupervised_save_epoch(
        self,
        epoch: int,
        val_stat: dir,
        test_out_stat: dir,
        config: Union[CommonArgs, Munch],
    ):
        r"""
        Training util for checkpoint saving.

        Returns:
            None

        """
        state_dict = self.model.state_dict()
        ckpt = {
            "state_dict": state_dict,
            "val_in_loss": val_stat["loss"],
            "test_out_stat_fpr": test_out_stat["fpr"],
            "test_out_stat_auroc": test_out_stat["auroc"],
            "time": datetime.datetime.now().strftime("%b%d %Hh %M:%S"),
            "model": {
                "model name": f"{config.model.model_name} {config.model.model_level} layers",
                "dim_hidden": config.model.dim_hidden,
                "dim_ffn": config.model.dim_ffn,
                "global pooling": config.model.global_pool,
            },
            "dataset": config.dataset.dataset_name,
            "train": {
                "weight_decay": config.train.weight_decay,
                "learning_rate": config.train.lr,
                "mile stone": config.train.mile_stones,
                "Batch size": f"{config.train.train_bs}, {config.train.val_bs}, {config.train.test_bs}",
            },
            "OOD": {
                "OOD alg": config.ood.ood_alg,
                "OOD param": config.ood.ood_param,
                "number of environments": config.dataset.num_envs,
            },
            "log file": config.log_path,
            "epoch": epoch,
            "max epoch": config.train.max_epoch,
        }
        if epoch < config.train.pre_train:
            return

        if not os.path.exists(os.path.join(config.ckpt_dir, config.task)):
            os.makedirs(os.path.join(config.ckpt_dir, config.task))
            print(
                f"#W#Directory does not exists. Have built it automatically.\n"
                f"{os.path.abspath(os.path.join(config.ckpt_dir, 'pretrain'))}"
            )

        saved_file = None
        if (
            config.metric.best_stat["loss"] is None
            or val_stat["loss"] <= config.metric.best_stat["loss"]
            or epoch % config.train.save_gap == 0
        ):
            saved_file = os.path.join(config.ckpt_dir, config.task, f"{epoch}.ckpt")
            torch.save(ckpt, saved_file)
            shutil.copy(
                saved_file, os.path.join(config.ckpt_dir, config.task, f"last.ckpt")
            )

        # --- Out-Of-Domain checkpoint ---
        if (
            config.metric.best_stat["loss"] is None
            or val_stat["loss"] <= config.metric.best_stat["loss"]
        ):
            config.metric.best_stat["loss"] = val_stat["loss"]
            shutil.copy(
                saved_file, os.path.join(config.ckpt_dir, config.task, f"best.ckpt")
            )
            print("#IM#Saved a new best checkpoint.")
            self.early_stop_counter = 0
        else:
            self.early_stop_counter += 1
            if self.early_stop_counter >= config.train.early_stop:
                if config.task == "train":
                    self.task = "test"
                    self.load_task()
                raise ValueError(f"Early Stopping.")
        if saved_file is not None and config.clean_save:
            os.unlink(saved_file)
