'''
Author: Chris Xiao yl.xiao@mail.utoronto.ca
Date: 2024-02-15 16:17:54
LastEditors: Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime: 2024-02-17 20:11:56
FilePath: /mbp1413-final/models/utils.py
Description: utility functions for the project
I Love IU
Copyright (c) 2024 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved. 
'''
import monai
from monai.data import DataLoader
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    RandAffined,
    Resized,
    Lambdad
)
import torch
import torch.nn as nn
import glob
import os
import gdown
import shutil
import zipfile
from typing import Tuple, Dict, Any, Sequence, Union
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from omegaconf import OmegaConf

ROOT = Path(os.path.dirname(os.path.realpath(__file__))).parent

def download_dataset(cfg: Dict[str, Any]) -> None:
    assert cfg.dataset.url is not None or cfg.dataset.url != "", "Please provide the URL of the dataset"
    shutil.rmtree(os.path.join(ROOT, 'datasets'), ignore_errors=True)
    gdown.download(url=cfg.dataset.url, output="dataset.zip", quiet=False, fuzzy=True)
    unzip_dataset("dataset.zip", os.path.join(ROOT, "datasets/raw"))
    os.remove("dataset.zip")


def make_if_dont_exist(
    folder_path: str,
    overwrite: bool = False
) -> None:
    if os.path.exists(folder_path):
        if not overwrite:
            pass
        else:
            shutil.rmtree(folder_path, ignore_errors = True)
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

def FullDiceScore() -> nn.Module:
    score = monai.metrics.DiceHelper(
        include_background=True,
        reduction="none"
    )
    return score

def FullJaccardLoss() -> nn.Module:
    score = monai.losses.DiceLoss(
        include_background=True,
        to_onehot_y=True,
        jaccard=True,
        reduction="none"
    )
    return score

def create_dirs() -> Tuple[str, str, str, str]:
    mapped_path = os.path.join(ROOT, "datasets/mapped")
    train_path = os.path.join(mapped_path, "train")
    test_path = os.path.join(mapped_path, "test")
    train_images_path = os.path.join(train_path, "images")
    train_masks_path = os.path.join(train_path, "masks")
    test_images_path = os.path.join(test_path, "images")
    test_masks_path = os.path.join(test_path, "masks")
    make_if_dont_exist(mapped_path, overwrite=True)
    make_if_dont_exist(train_path, overwrite=True)
    make_if_dont_exist(test_path, overwrite=True)
    make_if_dont_exist(train_images_path, overwrite=True)
    make_if_dont_exist(train_masks_path, overwrite=True)
    make_if_dont_exist(test_images_path, overwrite=True)
    make_if_dont_exist(test_masks_path, overwrite=True)
    return train_images_path, train_masks_path, test_images_path, test_masks_path

def map_dataset(
    cfg: Dict[str, Any]
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    stage = cfg.dataset.stage
    train_zip = os.path.join(ROOT, "datasets/raw", f"stage{stage}_train.zip")
    test_zip = os.path.join(ROOT, "datasets/raw", f"stage{stage}_test.zip")
    unzip_dataset(train_zip, os.path.join(ROOT, "datasets/raw", "stage1_train"))
    unzip_dataset(test_zip, os.path.join(ROOT, "datasets/raw", "stage1_test"))
    
    train_images_path, train_masks_path, test_images_path, test_masks_path = create_dirs()
    for i in os.listdir(os.path.join(ROOT, "datasets/raw", f"stage{stage}_train")):
        assert os.path.exists(os.path.join(ROOT, "datasets/raw", f"stage{stage}_train", i, "images")), "Images folder not found"
        assert os.path.exists(os.path.join(ROOT, "datasets/raw", f"stage{stage}_train", i, "masks")), "Masks folder not found"
        for j in sorted(glob.glob(os.path.join(ROOT, "datasets/raw", f"stage{stage}_train", i, "images", "*.png"))):
            shutil.copy(j, train_images_path)
        
        m = 0
        for k in sorted(glob.glob(os.path.join(ROOT, "datasets/raw", f"stage{stage}_train", i, "masks", "*.png"))):
            m += 1
            mask = sitk.GetArrayFromImage(sitk.ReadImage(k))
            mask[mask != 0] = 255
            if m == 1:
                combined_label = mask
                continue
            combined_label += mask
            
        sitk.WriteImage(sitk.GetImageFromArray(combined_label), os.path.join(train_masks_path, i + ".png"))
    
    for i in os.listdir(os.path.join(ROOT, "datasets/raw", f"stage{stage}_test")):
        assert os.path.exists(os.path.join(ROOT, "datasets/raw", f"stage{stage}_test", i, "images")), "Images folder not found"
        assert not os.path.exists(os.path.join(ROOT, "datasets/raw", f"stage{stage}_test", i, "masks")), "Masks folder found"
        for j in sorted(glob.glob(os.path.join(ROOT, "datasets/raw", f"stage{stage}_test", i, "images", "*.png"))):
            shutil.copy(j, test_images_path)
    
    return load_dataset(train_images_path, train_masks_path, test_images_path, test_masks_path, cfg)

def convert_to_greyscale(image):
    return image.convert('L')

def load_dataset(
    train_images_path: str,
    train_masks_path: str,
    test_images_path: str,
    test_masks_path: str,
    cfg: OmegaConf
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # create a dic with directory of images and masks
    train_files = [{"image": img, "label": mask} for img, mask in 
                   zip(sorted(glob.glob(os.path.join(train_images_path, "*.png"))), 
                       sorted(glob.glob(os.path.join(train_masks_path, "*.png"))))]
    test_files = [{"image": img, "label": mask} for img, mask in 
                  zip(sorted(glob.glob(os.path.join(test_images_path, "*.png"))), 
                      sorted(glob.glob(os.path.join(test_masks_path, "*.png"))))]
    # define transforms
    train_transforms = Compose(
        [
        # Load image and label data
        LoadImaged(keys=["image", "label"]), #pn files loaded as PIL image
        # Ensure channel is the first dimension
        EnsureChannelFirstd(keys=["image", "label"]),
        # Convert images and masks to grey scale
        Lambdad(keys=["image", "label"], func=convert_to_greyscale),
        # resize images and masks with scaling
        Resized(keys=["image", "label"], spatial_size=(256, 256), mode=("bilinear", "nearest")),
        # Scale intensity values of the image within the specified range
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-57,
            a_max=164,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        # Crop foreground from the image and label using the source image
        # so that it will not learn the background
        CropForegroundd(keys=["image", "label"], source_key="image"),
        # Adjust the orientation of the image and label using RAS (Right, Anterior, Superior) axcodes
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        # Adjust the spacing of the image and label using specified pixel dimensions and interpolation modes
        # this helps with data integration
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        # Randomly crop regions with positive and negative labels to create training samples
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96), # this will be the size of our cropped images to be inputted
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        # random transforms
        RandAffined(
            keys=['image', 'label'],
            mode=('bilinear', 'nearest'),
            prob=1.0, spatial_size=(96, 96, 96),
            rotate_range=(0, 0, np.pi/15),
            scale_range=(0.1, 0.1, 0.1))
        ]
    )

# Define validation and test transforms
    val_transforms = Compose(
        [
            # Load image and label data
            LoadImaged(keys=["image", "label"]),
            # Ensure channel is the first dimension
            EnsureChannelFirstd(keys=["image", "label"]),
            # Convert images and masks to grey scale
            Lambdad(keys=["image", "label"], func=convert_to_greyscale),
            # resize images and masks with scaling
            Resized(keys=["image", "label"], spatial_size=(256, 256), mode=("bilinear", "nearest")),
            # Scale intensity values of the image within the specified range
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            # Crop foreground from the image and label using the source image
            CropForegroundd(keys=["image", "label"], source_key="image"),
            # Adjust the orientation of the image and label using RAS (Right, Anterior, Superior) axcodes
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            # Adjust the spacing of the image and label using specified pixel dimensions and interpolation modes
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        ]
    )

    # load datasets
    tran_size = int(0.8 * len(train_files))
    val_size = len(train_files) - tran_size
    train_data, test_data = torch.utils.data.random_split(train_files, [tran_size, val_size])
    print(f"Training data size: {len(train_data)}")
    print(f"Validation data size: {len(test_data)}")

    tr_loader = DataLoader(
        train_data,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        transform=train_transforms
    )
    val_loader = DataLoader(
        test_data,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        transform=val_transforms
    )
    te_loader = DataLoader(
        test_files,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        transform=val_transforms
    )
    
    return tr_loader, val_loader, te_loader

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
