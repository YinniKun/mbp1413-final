'''
Author: Chris Xiao yl.xiao@mail.utoronto.ca
Date: 2024-02-15 16:17:54
LastEditors: Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime: 2024-02-15 21:47:51
FilePath: /mbp1413-final/models/utils.py
Description: utility functions for the project
I Love IU
Copyright (c) 2024 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved. 
'''
import monai
import torch
import torch.nn as nn
import glob
import os
import gdown
import shutil
import zipfile
from typing import List, Tuple, Dict, Any


def download_dataset(cfg: Dict[str, Any]) -> None:
    assert cfg.dataset.url is not None or cfg.dataset.url != "", "Please provide the URL of the dataset"

    raw_path = os.path.join(cfg.dataset.root_dir, "raw")
    make_if_dont_exist(cfg.dataset.root_dir)
    make_if_dont_exist(raw_path)
    gdown.download(url=cfg.dataset.url, output=os.path.join(
        raw_path, "dataset.zip"), quiet=False, fuzzy=True)
    unzip_dataset(os.path.join(raw_path, "dataset.zip"), raw_path)
    os.remove(os.path.join(raw_path, "dataset.zip"))


def make_if_dont_exist(folder_path: str) -> None:
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path, ignore_errors=True)
        os.makedirs(folder_path)
    else:
        os.makedirs(folder_path)


def unzip_dataset(
    zip_path: str,
    target_path: str
) -> None:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_path)


def DFLoss() -> nn.Module:
    loss = monai.losses.GeneralizedDiceFocalLoss(
        include_background=True,
        to_onehot_y=True,  # Added a comma here
        softmax=True,
        reduction="mean"
    )
    return loss

def DiceScore() -> nn.Module:
    score = monai.metrics.DiceHelper(
        include_background=True,
        reduction="none"
    )
    return score

def JaccardLoss() -> nn.Module:
    score = monai.losses.DiceLoss(
        include_background=True,
        to_onehot_y=True,
        jaccard=True,
        reduction="none"
    )
    return score

def preprocess_dataset(
    dataset: List[str],
    cfg: Dict[str, Any]
) -> Tuple[List[str], List[str]]:
    train_files = []
    val_files = []
    for i, file in enumerate(dataset):
        if i % 10 == 0:
            val_files.append(file)
        else:
            train_files.append(file)
    return train_files, val_files
