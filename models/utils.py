'''
Author: Chris Xiao yl.xiao@mail.utoronto.ca
Date: 2024-02-15 16:17:54
LastEditors: Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime: 2024-02-16 00:29:43
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
from typing import List, Tuple, Dict, Any, Sequence, Union
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

ROOT = "."

def download_dataset(cfg: Dict[str, Any]) -> None:
    assert cfg.dataset.url is not None or cfg.dataset.url != "", "Please provide the URL of the dataset"
    shutil.rmtree(os.path.join(ROOT, 'datasets'), ignore_errors=True)
    gdown.download(url=cfg.dataset.url, output="dataset.zip", quiet=False, fuzzy=True)
    unzip_dataset("dataset.zip", os.path.join(ROOT, "datasets/raw"))
    os.remove("dataset.zip")


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
        reduction="mean"
    )
    return score

def JaccardLoss() -> nn.Module:
    score = monai.losses.DiceLoss(
        include_background=True,
        to_onehot_y=True,
        jaccard=True,
        reduction="mean"
    )
    return score

def create_dirs() -> None:
    processed_path = os.path.join(ROOT, "datasets/processed")
    train_path = os.path.join(processed_path, "train")
    test_path = os.path.join(processed_path, "test")
    train_images_path = os.path.join(train_path, "images")
    train_masks_path = os.path.join(train_path, "masks")
    test_images_path = os.path.join(test_path, "images")
    test_masks_path = os.path.join(test_path, "masks")
    make_if_dont_exist(processed_path)
    make_if_dont_exist(train_path)
    make_if_dont_exist(test_path)
    make_if_dont_exist(train_images_path)
    make_if_dont_exist(train_masks_path)
    make_if_dont_exist(test_images_path)
    make_if_dont_exist(test_masks_path)
    return train_images_path, train_masks_path, test_images_path, test_masks_path

def remap_dataset(
    cfg: Dict[str, Any]
) -> None:
    
    stage = cfg.dataset.stage
    train_zip = os.path.join(ROOT, "datasets/raw", f"stage{stage}_train.zip")
    test_zip = os.path.join(ROOT, "datasets/raw", f"stage{stage}_test.zip")
    unzip_dataset(train_zip, os.path.join(ROOT, "datasets/raw", "stage1_train"))
    unzip_dataset(test_zip, os.path.join(ROOT, "datasets/raw", "stage1_test"))
    
    train_images_path, train_masks_path, test_images_path, test_masks_path = create_dirs()
    for i in os.listdir(os.path.join(ROOT, "datasets/raw", f"stage{stage}_train")):
        assert os.path.exists(os.path.join(ROOT, "datasets/raw", f"stage{stage}_train", i, "images")), "images folder not found"
        assert os.path.exists(os.path.join(ROOT, "datasets/raw", f"stage{stage}_train", i, "masks")), "masks folder not found"
        for j in sorted(glob.glob(os.path.join(ROOT, "datasets/raw", f"stage{stage}_train", i, "images", "*.png"))):
            shutil.copy(j, train_images_path)
        
        m = 1
        for k in sorted(glob.glob(os.path.join(ROOT, "datasets/raw", f"stage{stage}_train", i, "masks", "*.png"))):
            mask = sitk.GetArrayFromImage(sitk.ReadImage(k))
            mask[mask != 0] = 255
            if m == 1:
                combined_label = mask
                pass
            combined_label += mask
            m += 1
        
        sitk.WriteImage(sitk.GetImageFromArray(combined_label), os.path.join(train_masks_path, i + ".png"))
    
    for i in os.listdir(os.path.join(ROOT, "datasets/raw", f"stage{stage}_test")):
        assert os.path.exists(os.path.join(ROOT, "datasets/raw", f"stage{stage}_test", i, "images")), "images folder not found"
        assert not os.path.exists(os.path.join(ROOT, "datasets/raw", f"stage{stage}_test", i, "masks")), "masks folder found"
        for j in sorted(glob.glob(os.path.join(ROOT, "datasets/raw", f"stage{stage}_test", i, "images", "*.png"))):
            shutil.copy(j, test_images_path)


def load_dataset(
    train_images_path: str,
    test_images_path: str
) -> None:
    pass

def plot_progress(
    save_dir: str, 
    train_losses: Sequence[Sequence[Union[int, float]]], 
    val_losses: Sequence[Sequence[Union[int, float]]], 
    dice_scores: Sequence[Sequence[Union[int, float]]],
    iou_scores: Sequence[Sequence[Union[int, float]]],
    name: str
) -> None:
    """
    Should probably by improved
    :return:
    """
    assert len(train_losses) != 0
    train_losses = np.array(train_losses)
    try:
        font = {'weight': 'normal',
                'size': 18}

        matplotlib.rc('font', **font)

        fig = plt.figure(figsize=(30, 24))
        ax = fig.add_subplot(111)
        ax1 = ax.twinx()
        ax.plot(train_losses[:,0], train_losses[:,1], color='b', ls='-', label="loss_tr")
        if len(val_losses) != 0:
            dice_scores = np.array(dice_scores)
            iou_scores = np.array(iou_scores)
            val_losses = np.array(val_losses)
            ax.plot(val_losses[:, 0], val_losses[:, 1], color='r', ls='-', label="loss_val")
            ax.plot(dice_scores[:, 0], dice_scores[:, 1], color='g', ls='-', label="dsc_val")
            ax.plot(iou_scores[:, 0], iou_scores[:, 1], color='purple', ls='-', label="iou_val")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax1.set_ylabel("score")
        ax.legend()
        ax.set_title(name)
        fig.savefig(os.path.join(save_dir, name + ".png"))
        plt.cla()
        plt.close(fig)
    except:
        print(f"failed to plot {name} training progress")
