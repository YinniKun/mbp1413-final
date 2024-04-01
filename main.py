'''
Author: Chris Xiao yl.xiao@mail.utoronto.ca
Date: 2024-02-15 16:24:56
LastEditors: Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime: 2024-03-31 23:17:31
FilePath: /mbp1413-final/main.py
Description: main script for the project
I Love IU
Copyright (c) 2024 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved. 
'''
from omegaconf import OmegaConf
import argparse
import os
from models import download_dataset, load_dataset, check_device
from models import unet
from models import unetr
import torch
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.realpath(__file__)))

modules = {
    "unet": unet,
    "unetr": unetr
}


def parse_command() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MBP1413 Final Project")
    parser.add_argument("-c", "--cfg", type=str,
                        default="config.yaml", help="path to the config file")
    parser.add_argument("-mo", "--model", type=str, default="unet",
                        help="model to use, either unet or unetr")
    parser.add_argument("-m", "--mode", type=str, default="train",
                        help="mode of the program, either test or train")
    parser.add_argument("-d", "--download", action="store_true",
                        help="use this if you want to download the dataset")
    parser.add_argument("-r", "--resume", action="store_true",
                        help="use this if you want to continue a training")
    parser.add_argument("-e", "--epochs", type=int,
                        default=200, help="Number of epochs for training")
    parser.add_argument("-l", "--learning_rate", type=float,
                        default=0.005, help="Learning rate")
    parser.add_argument("-opt", "--optimizer", type=str,
                        default="Adam", help="Optimizer of model, default is Adam")
    parser.add_argument("-sch", "--scheduler", action="store_true",
                        help="Use this parameter to use lr scheduler")
    parser.add_argument("-s", "--save", action="store_true",
                        help="Use this if you want to save the architecture plot")
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
    device = check_device(cfg)
    model_name = args.model.lower()
    if model_name in modules.keys():
        model = modules[model_name](cfg, args.learning_rate, args.epochs, device,
                                    args.model, args.optimizer, args.scheduler,
                                    tr_loader, val_loader, te_loader)
    else:
        raise ValueError("model not supported")

    if args.mode == "train":
        print(f"Training {args.model}")
        if args.resume:
            model.load_checkpoint(mode="last")
        else:
            model.init_training_dir()
        
        if args.save:
            model.plot_architecture(mode='train')

        model.train()
        print(f"Training completed for {args.model}")
    elif args.mode == "test":
        print(f"Testing {args.model}")
        model.init_inference_dir()
        if args.save:
            model.plot_architecture(mode='test')
        model.test()
        print(f"Testing completed for {args.model}")
    else:
        raise ValueError("mode not supported")
    


if __name__ == "__main__":
    main()
