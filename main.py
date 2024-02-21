'''
Author: Chris Xiao yl.xiao@mail.utoronto.ca
Date: 2024-02-15 16:24:56
LastEditors: Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime: 2024-02-21 03:04:18
FilePath: /mbp1413-final/main.py
Description: main script for the project
I Love IU
Copyright (c) 2024 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved. 
'''
from omegaconf import OmegaConf
import argparse
import os
from models.utils import download_dataset, map_dataset
from models.unet import unet
from models.unetr import unetr
import torch

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
    # Done: add data augmentation and dataloader
    tr_loader, val_loader, te_loader = map_dataset(cfg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # use GPU if available
    # device = torch.device("cpu")
    model = modules[cfg.model.name](cfg, device, tr_loader, val_loader, te_loader)
    
    if args.mode == "train":
        if args.resume:
            model.load_checkpoint(mode="last")
        else:
            model.init_training_dir()
        
        model.train()

    elif args.mode == "test":
        # TODO: find test masks
        model.init_inference_dir()
        model.test()
    else:
        raise ValueError("mode not supported")

if __name__ == "__main__":
    main()

