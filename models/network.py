'''
Author: Chris Xiao yl.xiao@mail.utoronto.ca
Date: 2024-02-15 14:52:45
LastEditors: Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime: 2024-03-31 23:16:02
FilePath: /mbp1413-final/models/network.py
Description: base network class for the project
I Love IU
Copyright (c) 2024 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved. 
'''

import torch
import torch.nn as nn
from torchviz import make_dot
from .utils import *
from typing import Dict, Any, Tuple, List
import monai
from monai.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import json
from PIL import Image
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.realpath(__file__))).parent


class Network(nn.Module):
    def __init__(
        self,
        cfg: Dict[str, Any],
        lr: float,
        epoch: int,
        device: torch.device,
        name: str,
        optimizer: str,
        use_sche: bool,
        tr_loader: DataLoader,
        val_loader: DataLoader,
        te_loader: DataLoader
    ) -> None:
        super(Network, self).__init__()
        self.cfg = cfg
        self.lr = lr
        self.max_epoch = epoch
        self.device = device
        self.tr_loader = tr_loader
        self.val_loader = val_loader
        self.te_loader = te_loader
        self.use_sche = use_sche
        self.optim = optimizer

        train_folder_name = define_name(
            name, lr, epoch, optimizer, use_sche, 'train')
        # Training save dirs
        self.training_dir = os.path.join(ROOT, train_folder_name)
        if self.cfg.training.save_dir != "":
            self.training_dir = self.cfg.training.save_dir

        self.weights_dir = os.path.join(self.training_dir, 'weights')
        self.plots_dir = os.path.join(self.training_dir, 'plots')
        # Inference save dirs
        test_folder_name = define_name(
            name, lr, epoch, optimizer, use_sche, 'test')
        self.inference_dir = os.path.join(ROOT, test_folder_name)
        if self.cfg.inference.predict_dir != "":
            self.inference_dir = self.cfg.inference.predict_dir

    def init_model(self) -> None:
        pass

    def init_training_dir(self) -> None:
        # Create save directory
        make_if_dont_exist(self.training_dir, overwrite=True)
        make_if_dont_exist(self.weights_dir, overwrite=True)
        make_if_dont_exist(self.plots_dir, overwrite=True)

    def init_inference_dir(self) -> None:
        # Create save directory
        make_if_dont_exist(self.inference_dir, overwrite=True)

    def train(self) -> None:
        for epoch in range(self.start_epoch, self.max_epoch):
            self.epoch = epoch
            self.model.train()
            tr_loss = []
            with tqdm(self.tr_loader, unit="batch") as tepoch:
                for batch in tepoch:
                    tepoch.set_description(
                        f"Epoch {epoch+1}/{self.max_epoch} Training")
                    self.optimizer.zero_grad()
                    x, y = batch["image"].to(
                        self.device), batch["label"].to(self.device)
                    y[y != 0] = 1  # convert to binary mask
                    y_pred = self.model(x)
                    loss = self.loss(y_pred, y)
                    loss.backward()
                    self.optimizer.step()
                    tr_loss.append(loss.item())
                    if self.use_sche:
                        tepoch.set_postfix(
                            loss=loss.item(), lr=self.lr_scheduler.get_last_lr()[-1])
                    else:
                        tepoch.set_postfix(loss=loss.item(), lr=self.lr)

            if self.use_sche:
                self.lr_scheduler.step(self.epoch)
            self.train_losses.append([self.epoch+1, np.mean(tr_loss, axis=0)])

            if (self.epoch+1) % self.valid_period == 0:
                val_loss = []
                val_dsc = []
                val_iou = []
                self.model.eval()
                with torch.no_grad():
                    with tqdm(self.val_loader, unit='batch') as tepoch:
                        for batch in tepoch:
                            tepoch.set_description(
                                f"Epoch {epoch+1}/{self.max_epoch} Validation")
                            x, y = batch["image"].to(
                                self.device), batch["label"].to(self.device)
                            y[y != 0] = 1  # convert to binary mask
                            y_pred = self.model(x)
                            y_pred_logits = torch.argmax(torch.softmax(
                                y_pred, dim=1), dim=1, keepdim=True)
                            y_pred_class = monai.networks.utils.one_hot(
                                y_pred_logits, num_classes=self.cfg.model.out_channels)
                            loss = self.loss(y_pred, y)
                            dice_score, iou_score = self.score(y_pred_class, y)
                            val_dsc.append(dice_score.item())
                            val_iou.append(iou_score.item())
                            val_loss.append(loss.item())
                            tepoch.set_postfix(
                                loss=loss.item(), dice=dice_score.item(), iou=iou_score.item())

                self.valid_losses.append(
                    [self.epoch+1, np.mean(val_loss, axis=0)])
                self.dscs.append([self.epoch+1, np.mean(val_dsc, axis=0)])
                self.ious.append([self.epoch+1, np.mean(val_iou, axis=0)])

                if np.mean(val_loss, axis=0) < self.best_valid_loss:
                    self.best_valid_loss = np.mean(val_loss, axis=0)
                    self.save_checkpoint(mode="best")

            self.save_checkpoint(mode="last")
            plot_progress(self.plots_dir, self.train_losses,
                          self.valid_losses, self.dscs, self.ious, "loss")

    def test(self) -> None:
        results = {}
        self.load_checkpoint(mode="best")
        self.model.eval()
        dscs = []
        ious = []
        with torch.no_grad():
            with tqdm(self.te_loader, unit='batch') as tepoch:
                for batch in tepoch:
                    filename = os.path.basename(
                        batch["image_meta_dict"]["filename_or_obj"][0])
                    tepoch.set_description("Model Inference")
                    x, y = batch["image"].to(
                        self.device), batch["label"].to(self.device)
                    y[y != 0] = 1  # convert to binary mask
                    y_pred = self.model(x)
                    y_pred_label = torch.argmax(torch.softmax(
                        y_pred, dim=1), dim=1, keepdim=True)
                    y_pred_label_numpy = y_pred_label.detach(
                    ).cpu().numpy().astype(np.uint8)[0, 0, ...]
                    y_pred_label_numpy[y_pred_label_numpy != 0] = 255
                    seg = Image.fromarray(y_pred_label_numpy)
                    seg.save(os.path.join(self.inference_model_dir, filename))
                    y_pred_class = monai.networks.utils.one_hot(
                        y_pred_label, num_classes=self.cfg.model.out_channels)
                    dice_score, iou_score = self.full_score(y_pred_class, y)

                    list_dsc = dice_score.detach().cpu().numpy().astype(np.float64).tolist()
                    list_iou = iou_score.detach().cpu().numpy().astype(np.float64).tolist()
                    dscs.append(list_dsc)
                    ious.append(list_iou)
                    results[filename] = {
                        "Dice Score": list_dsc,
                        "IoU": list_iou
                    }

        avg_dsc = np.mean(dscs, axis=0)
        avg_iou = np.mean(ious, axis=0)
        results['Average'] = {
            "Dice Score": avg_dsc.tolist(),
            "IoU": avg_iou.tolist()
        }
        with open(os.path.join(self.inference_model_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=4)

        f.close()

    def loss(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        return self.loss_metric(y_pred, y)

    def score(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.dice_metric(y_pred, y)[0], 1-self.jaccard_loss(y_pred, y)

    def full_score(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        return self.full_dice_metric(y_pred, y)[0].flatten(), 1-(self.full_jaccard_loss(y_pred, y).flatten())

    def init_params(self) -> None:
        self.loss_metric = DFLoss()
        self.dice_metric = DiceScore()
        self.jaccard_loss = JaccardLoss()
        self.full_dice_metric = FullDiceScore()
        self.full_jaccard_loss = FullJaccardLoss()
        if self.optim == "SGD":
            # ! Switch to SGD optimizer
            self.optimizer = getattr(torch.optim, "SGD")(
                self.model.parameters(), lr=self.lr, weight_decay=3e-5, momentum=0.99, nesterov=True)
        else:
            self.optimizer = getattr(torch.optim, "Adam")(
                self.model.parameters(), lr=self.lr, weight_decay=3e-5)
        if self.use_sche:
            # ! Adapt learning scheduler from nnUnet-v2
            self.lr_scheduler = PolyLRScheduler(
                self.optimizer, self.lr, self.max_epoch)
        self.train_losses, self.valid_losses, self.dscs, self.ious = [], [], [], []
        self.best_valid_loss = np.inf
        self.valid_period = self.cfg.training.val_period
        self.start_epoch, self.epoch = 0, 0

    def load_checkpoint(
        self,
        mode: str = None,
    ) -> None:

        assert os.path.exists(os.path.join(
            self.weights_dir, f'{mode}_ckpt.pth')), f"{mode.capitalize()} checkpoint not found !!!"
        ckpt = torch.load(os.path.join(self.weights_dir,
                          f'{mode}_ckpt.pth'), map_location=self.device)

        # if ckpt_path == "":
        #     assert os.path.exists(os.path.join(self.weights_dir, f'{mode}_ckpt.pth')), f"{mode.capitalize()} checkpoint not found !!!"
        #     ckpt = torch.load(os.path.join(self.weights_dir, f'{mode}_ckpt.pth'), map_location=self.device)
        # else:
        #     assert os.path.exists(ckpt_path), "Checkpoint not found !!!"
        #     ckpt = torch.load(ckpt_path, map_location=self.device)

        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.model.load_state_dict(ckpt['weights'])
        if self.use_sche:
            self.lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        self.best_valid_loss = ckpt['best_valid_loss']
        self.train_losses = ckpt['train_losses']
        self.valid_losses = ckpt['valid_losses']
        self.lr = self.optimizer.param_groups[0]['lr']
        self.start_epoch = ckpt['epoch'] + 1
        self.dscs = ckpt['valid_dsc']
        self.ious = ckpt['valid_iou']

    def save_checkpoint(
        self,
        mode: str = None
    ) -> None:
        save_path = os.path.join(self.weights_dir, f'{mode}_ckpt.pth')
        if not self.use_sche:
            torch.save(
                {
                    'epoch': self.epoch,
                    'weights': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'best_valid_loss': self.best_valid_loss,
                    'train_losses': self.train_losses,
                    'valid_losses': self.valid_losses,
                    "valid_dsc": self.dscs,
                    "valid_iou": self.ious
                }, save_path
            )
        else:
            torch.save(
                {
                    'epoch': self.epoch,
                    'weights': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'lr_scheduler': self.lr_scheduler.state_dict(),
                    'best_valid_loss': self.best_valid_loss,
                    'train_losses': self.train_losses,
                    'valid_losses': self.valid_losses,
                    "valid_dsc": self.dscs,
                    "valid_iou": self.ious
                }, save_path
            )

    def plot_architecture(
        self, 
        mode: str
    ) -> None:
        x = next(iter(self.tr_loader))["image"][[0]]
        y = self.model(x.to(self.device))
        if mode == 'train':
        # make_dot(y.mean(), params=dict(self.model.named_parameters()), show_attrs=True, show_saved=True).render(
        #     filename="network", directory=self.training_dir, format="png"
        # )
            save_dir = self.training_dir
        else:
            save_dir = self.inference_dir
        
        make_dot(y.mean(), params=dict(self.model.named_parameters()), show_attrs=True, show_saved=True).render(
            filename="network", directory=save_dir, format="png"
        )

        
        
        
