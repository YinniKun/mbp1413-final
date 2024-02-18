'''
Author: Chris Xiao yl.xiao@mail.utoronto.ca
Date: 2024-02-15 14:52:58
LastEditors: Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime: 2024-02-17 19:03:12
FilePath: /mbp1413-final/models/unetr.py
Description: transformer-based U-Net model for medical image segmentation
I Love IU
Copyright (c) 2024 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved. 
'''
import monai
import torch
import torch.nn as nn
from .network import Network
from typing import Dict, Any
from monai.data import DataLoader


class unetr(Network):
    def __init__(
        self, 
        cfg: Dict[str, Any],
        device: torch.device,
        tr_loader: DataLoader,
        val_loader: DataLoader,
        te_loader: DataLoader
    ) -> None:
        super(unetr, self).__init__(cfg, device, tr_loader, val_loader, te_loader)
        self.init_model()
        self.init_params()
    
    def init_model(self) -> None:
        # Create U-Net model
        self.model = monai.networks.nets.UNETR(
            spatial_dim=self.cfg.model.spatial_dim,
            in_channels=self.cfg.model.in_channels,
            out_channels=self.cfg.model.out_channels,
            image_size=self.cfg.model.image_size,
            channels=self.cfg.model.channels,
            strides=self.cfg.model.strides,
            kernel_size=self.cfg.model.kernel_size,
            num_res_units=self.cfg.model.num_res_units,
            act=self.cfg.model.activation,
            norm=self.cfg.model.normalization,
            dropout=self.cfg.model.dropout,
        ).to(self.device)