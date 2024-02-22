'''
Author: Chris Xiao yl.xiao@mail.utoronto.ca
Date: 2024-02-15 16:24:56
LastEditors: Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime: 2024-02-21 20:01:12
FilePath: /mbp1413-final/main.py
Description: main script for the project
I Love IU
Copyright (c) 2024 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved. 
'''
from omegaconf import OmegaConf
import argparse
import os
from models.utils import download_dataset, load_dataset
from models.unet import unet
from models.unetr import unetr
import torch
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.realpath(__file__)))

modules = {
    "unet": unet,
    "unetr": unetr
}

def parse_command() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MBP1413 Final Project")
    parser.add_argument("-c", "--cfg", type=str, default="config.yaml", help="path to the config file")
    parser.add_argument("-m", "--mode", type=str, default="train", help="mode of the program")
    parser.add_argument("-d", "--download", action="store_true", help="use this if you want to download the dataset")
    parser.add_argument("-r", "--resume", action="store_true", help="use this if you want to continue a training")
    return parser.parse_args()

def main() -> None:
    args = parse_command()
    assert os.path.exists(args.cfg), "config file not found"
    cfg = OmegaConf.load(args.cfg)
    if args.download:
        download_dataset(cfg)
    train_path = os.path.join(ROOT, "datasets", "train")
    test_path = os.path.join(ROOT, "datasets", "test")
    tr_loader, val_loader, te_loader = load_dataset(train_path, test_path, cfg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # use GPU if available
    model = modules[cfg.model.name](cfg, device, tr_loader, val_loader, te_loader)
    if args.mode == "train":
        if args.resume:
            model.load_checkpoint(mode="last")
        else:
            model.init_training_dir()
        model.train()
    elif args.mode == "test":
        model.init_inference_dir()
        model.test()
    else:
        raise ValueError("mode not supported")

if __name__ == "__main__":
    main()

