'''
Author: Chris Xiao yl.xiao@mail.utoronto.ca
Date: 2024-02-15 14:52:45
LastEditors: Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime: 2024-02-15 22:55:53
FilePath: /mbp1413-final/models/unet.py
Description: base network class for the project
I Love IU
Copyright (c) 2024 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved. 
'''

import torch
import torch.nn as nn
from .utils import DFLoss, DiceScore, JaccardLoss, plot_progress
from typing import Dict, Any, Tuple
import monai
from monai.data import DataLoader
from tqdm import tqdm
import numpy as np
import os


class Network(nn.Module):
    def __init__(
        self,
        cfg: Dict[str, Any],
        device: torch.device,
    ) -> None:
        super(Network, self).__init__()
        self.cfg = cfg
        self.device = device

    def init_model(self):
        pass

    def train(
        self,
        tr_loader: DataLoader,
        val_loader: DataLoader
    ) -> None:
        for epoch in range(self.start_epoch, self.max_epoch):
            self.epoch = epoch
            self.model.train()
            tr_loss = []
            with tqdm(tr_loader, unit="batch") as tepoch:
                for batch in tepoch:
                    tepoch.set_description(
                        f"Epoch {epoch+1}/{self.cfg.training.epochs} Training")
                    self.optimizer.zero_grad()
                    x, y = batch["image"].to(
                        self.device), batch["label"].to(self.device)
                    y_pred = self.model(x)
                    loss = self.loss(y_pred, y)
                    loss.backward()
                    self.optimizer.step()
                    tr_loss.append(loss.item())
                    tepoch.set_postfix(loss=loss.item())

            self.train_losses.append([self.epoch+1, np.mean(tr_loss, axis=0)])

        if (self.epoch+1) % self.valid_period == 0:
            val_loss = []
            val_dsc = []
            val_iou = []
            self.model.eval()
            with torch.no_grad():
                with tqdm(val_loader, unit='batch') as tepoch:
                    for batch in tepoch:
                        tepoch.set_description(
                            f"Epoch {epoch+1}/{self.cfg.training.epochs} Validation")
                        x, y = batch["image"].to(
                            self.device), batch["label"].to(self.device)
                        y_pred = self.model(x)
                        y_pred_logits = torch.argmax(torch.softmax(
                            y_pred, dim=1), dim=1, keep_dim=True)
                        y_pred_class = monai.networks.utils.one_hot(
                            y_pred_logits, num_classes=self.cfg.model.out_channels)
                        loss = self.loss(y_pred, y)
                        dice_score, iou_score = self.score(y_pred_class, y)
                        val_dsc.append(dice_score.item())
                        val_iou.append(iou_score.item())
                        val_loss.append(loss.item())
                        tepoch.set_postfix(
                            loss=loss.item(), dice=dice_score.item(), iou=iou_score.item())

            self.valid_losses.append([self.epoch+1, np.mean(val_loss, axis=0)])
            self.dscs.append([self.epoch+1, np.mean(val_dsc, axis=0)])
            self.ious.append([self.epoch+1, np.mean(val_iou, axis=0)])

            if np.mean(val_loss, axis=0) < self.best_valid_loss:
                self.best_valid_loss = np.mean(val_loss, axis=0)
                self.save_checkpoint(best=True)

        self.save_checkpoint()
        plot_progress(os.path.join(self.cfg.training.save_dir, 'plots'),
                      self.train_losses, self.valid_losses, self.dscs, self.ious, "loss")

    def test(self) -> None:
        pass

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
        return self.dice_metric(y_pred, y), 1-self.jaccard_loss(y_pred, y)

    def init_params(self) -> None:
        self.loss_metric = DFLoss()
        self.lr = self.cfg.training.lr
        self.dice_metric = DiceScore()
        self.jaccard_loss = JaccardLoss()
        self.optimizer = getattr(torch.optim, self.cfg.training.optimizer)(
            self.model.parameters(), lr=self.lr)
        self.train_losses, self.valid_losses, self.dscs, self.ious = [], [], [], []
        self.best_valid_loss = np.inf
        self.max_epoch = self.cfg.training.epochs
        self.valid_period = self.cfg.training.val_period
        self.start_epoch, self.epoch = 0, 0

    def load_checkpoint(self) -> None:
        ckpt = torch.load(os.path.join(self.cfg.training.save_dir,
                          'weights/last_ckpt.pth'), map_location=self.device)
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.model.load_state_dict(ckpt['weights'])
        self.best_valid_loss = ckpt['best_valid_loss']
        self.train_losses = ckpt['train_losses']
        self.valid_losses = ckpt['valid_losses']
        self.lr = self.optimizer.param_groups[0]['lr']
        self.start_epoch = ckpt['epoch'] + 1
        self.dscs = ckpt['valid_dsc']
        self.ious = ckpt['valid_iou']

    def save_checkpoint(
        self,
        best: bool = False
    ) -> None:
        save_path = os.path.join(
            self.cfg.training.save_dir, 'weights/last_ckpt.pth')
        if best:
            save_path = os.path.join(
                self.cfg.training.save_dir, 'weights/best_ckpt.pth')
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
