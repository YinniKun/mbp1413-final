'''
Author: Chris Xiao yl.xiao@mail.utoronto.ca
Date: 2024-02-15 16:24:56
LastEditors: Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime: 2024-02-29 00:20:07
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
import gc

ROOT = Path(os.path.dirname(os.path.realpath(__file__)))

modules = {
    "unet": unet,
    "unetr": unetr
}

def parse_command() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MBP1413 Final Project")
    parser.add_argument("-c", "--cfg", type=str, default="config.yaml", help="path to the config file")
    parser.add_argument("-mo", "--model", type=str, default="unet", help="model to use, either unet or unetr")
    parser.add_argument("-m", "--mode", type=str, default="train", help="mode of the program, either test or train")
    parser.add_argument("-d", "--download", action="store_true", help="use this if you want to download the dataset")
    parser.add_argument("-r", "--resume", action="store_true", help="use this if you want to continue a training")
    parser.add_argument("-e", "--epochs", type=int, default=200, help="Number of epochs for training")
    parser.add_argument("-l", "--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("-t", "--title", type=str, default="Model", help="Title of the model")
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
    
    if args.models in modules.keys():
        model = modules[args.models](cfg, args.learning_rate, args.epochs, device, args.models, tr_loader, val_loader, te_loader)
    else:
        raise ValueError("model not supported")
    
    if args.mode == "train":
        print(f"Training {args.model}")
        if args.resume:
            model.load_checkpoint(mode="last")
        else:
            model.init_training_dir()
        model.train()
        print(f"Training completed for {args.model}")
    elif args.mode == "test":
        print(f"Testing {args.model}")
        model.init_inference_dir()
        model.test()
        print(f"Testing completed for {args.model}")
    else:
        raise ValueError("mode not supported")

if __name__ == "__main__":
    main()

