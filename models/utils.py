"""
Author: Chris Xiao yl.xiao@mail.utoronto.ca
Date: 2024-02-15 16:17:54
LastEditors: Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime: 2024-03-31 19:09:27
FilePath: /mbp1413-final/models/utils.py
Description: utility functions for the project
I Love IU
Copyright (c) 2024 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved.
"""

__all__ = [
    "download_dataset",
    "load_dataset",
    "check_device",
    "define_name",
    "make_if_dont_exist",
    "unzip_dataset",
    "DFLoss",
    "DiceScore",
    "JaccardLoss",
    "FullDiceScore",
    "FullJaccardLoss",
    "plot_progress",
    "PolyLRScheduler",
]
from pathlib import Path
import numpy as np
import monai
from monai.data import DataLoader, CacheDataset
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    ScaleIntensityRanged,
    Resized,
)
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
import glob
import os
import gdown
import shutil
import zipfile
from typing import Tuple, Dict, Any, Sequence, Union
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

ROOT = Path(os.path.dirname(os.path.realpath(__file__))).parent


def download_dataset(cfg: Dict[str, Any]) -> None:
    assert (
        cfg.dataset.url is not None or cfg.dataset.url != ""
    ), "Please provide the URL of the dataset"
    shutil.rmtree(os.path.join(ROOT, "datasets"), ignore_errors=True)
    gdown.download(url=cfg.dataset.url, output="dataset.zip", quiet=False, fuzzy=True)
    unzip_dataset("dataset.zip", ROOT)
    os.remove("dataset.zip")


def define_name(
    model_name: str,
    learning_rate: float,
    epoch: int,
    optimizer: str,
    scheduler: bool,
    mode: str,
) -> str:
    if mode == "train":
        ret = "training_"
    elif mode == "test":
        ret = "inference_"
    else:
        raise ValueError("mode not supported")
    ret += f"{model_name}_{learning_rate}_{epoch}_{optimizer}"
    if scheduler:
        ret += "_Sche"

    return ret


def make_if_dont_exist(folder_path: str, overwrite: bool = False) -> None:
    if os.path.exists(folder_path):
        if not overwrite:
            pass
        else:
            shutil.rmtree(folder_path, ignore_errors=True)
            os.makedirs(folder_path)
    else:
        os.makedirs(folder_path)


def unzip_dataset(zip_path: str, target_path: str) -> None:
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target_path)


def DFLoss() -> nn.Module:
    loss = monai.losses.GeneralizedDiceFocalLoss(
        include_background=True, to_onehot_y=True, softmax=True, reduction="mean"
    )
    return loss


def DiceScore() -> nn.Module:
    score = monai.metrics.DiceHelper(include_background=True, reduction="mean")
    return score


def JaccardLoss() -> nn.Module:
    score = monai.losses.DiceLoss(
        include_background=True, to_onehot_y=True, jaccard=True, reduction="mean"
    )
    return score


def FullDiceScore() -> nn.Module:
    score = monai.metrics.DiceHelper(include_background=True, reduction="none")
    return score


def FullJaccardLoss() -> nn.Module:
    score = monai.losses.DiceLoss(
        include_background=True, to_onehot_y=True, jaccard=True, reduction="none"
    )
    return score


def load_dataset(
    train_path: str, test_path: str, cfg: Dict[str, Any]
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # create a dic with directory of images and masks
    train_files = [
        {"image": img, "label": mask}
        for img, mask in zip(
            sorted(glob.glob(os.path.join(train_path, "images/*.png"))),
            sorted(glob.glob(os.path.join(train_path, "masks/*.png"))),
        )
    ]
    test_files = [
        {"image": img, "label": mask}
        for img, mask in zip(
            sorted(glob.glob(os.path.join(test_path, "images/*.png"))),
            sorted(glob.glob(os.path.join(test_path, "masks/*.png"))),
        )
    ]
    # define transforms
    transforms = Compose(
        [
            # png files loaded as PIL image, image is RGBA and mask is Grayscale
            LoadImaged(keys=["image", "label"], image_only=False, reader="PILReader"),
            # Ensure channel is the first dimension
            EnsureChannelFirstd(keys=["image", "label"]),
            # resize images and masks with scaling
            Resized(
                keys=["image", "label"],
                spatial_size=(512, 512),
                mode=("linear", "nearest-exact"),
            ),
            # Scale intensity values of the image within the specified range
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
        ]
    )
    # split & load datasets
    tran_size = int(0.8 * len(train_files))
    val_size = len(train_files) - tran_size
    train_data, val_data = torch.utils.data.random_split(
        train_files, [tran_size, val_size]
    )
    print(f"Training data size: {len(train_data)}")
    print(f"Validation data size: {len(val_data)}")

    tr_loader = DataLoader(
        CacheDataset(train_data, transform=transforms, cache_num=16, hash_as_key=True),
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
    )
    val_loader = DataLoader(
        CacheDataset(val_data, transform=transforms, cache_num=16, hash_as_key=True),
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )
    te_loader = DataLoader(
        CacheDataset(test_files, transform=transforms, cache_num=16, hash_as_key=True),
        batch_size=1,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )

    return tr_loader, val_loader, te_loader


def plot_progress(
    save_dir: str,
    train_losses: Sequence[Sequence[Union[int, float]]],
    val_losses: Sequence[Sequence[Union[int, float]]],
    dice_scores: Sequence[Sequence[Union[int, float]]],
    iou_scores: Sequence[Sequence[Union[int, float]]],
    name: str,
) -> None:
    """
    Should probably by improved
    :return:
    """
    assert len(train_losses) != 0
    train_losses = np.array(train_losses)
    try:
        font = {"weight": "normal", "size": 18}

        matplotlib.rc("font", **font)

        fig = plt.figure(figsize=(30, 24))
        ax = fig.add_subplot(111)
        ax1 = ax.twinx()
        ax.plot(
            train_losses[:, 0], train_losses[:, 1], color="b", ls="-", label="loss_tr"
        )
        if len(val_losses) != 0:
            dice_scores = np.array(dice_scores)
            iou_scores = np.array(iou_scores)
            val_losses = np.array(val_losses)
            ax.plot(
                val_losses[:, 0], val_losses[:, 1], color="r", ls="-", label="loss_val"
            )
            ax.plot(
                dice_scores[:, 0], dice_scores[:, 1], color="g", ls="-", label="dsc_val"
            )
            ax.plot(
                iou_scores[:, 0],
                iou_scores[:, 1],
                color="purple",
                ls="-",
                label="iou_val",
            )
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax1.set_ylabel("score")
        ax.legend()
        ax.set_title(name)
        fig.savefig(os.path.join(save_dir, name + ".png"))
        plt.cla()
        plt.close(fig)

    except Exception as e:
        print(f"failed to plot {name} training progress: {str(e)}")


def check_device(cfg: Dict[str, Any]) -> torch.device:
    """_summary_

    Args:
        cfg (Dict[str, Any]): configuration

    Returns:
        torch.device: device for training either cuda for nvidia gpu, mps for Apple mps or cpu
    """
    if cfg.training.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif cfg.training.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device


class PolyLRScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim,
        initial_lr: float,
        max_steps: int,
        exponent: float = 0.9,
        current_step: int = None,
    ) -> None:
        """_summary_

        Args:
            optimizer (torch.optim): training optimizer
            initial_lr (float): initial learning rate
            max_steps (int): number of training epochs
            exponent (float, optional): exponential value to decrease learning rate. Defaults to 0.9.
            current_step (int, optional): current epoch. Defaults to None.
        """
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1)

    def step(self, current_step: int = None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
