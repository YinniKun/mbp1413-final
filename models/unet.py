'''
Author: Chris Xiao yl.xiao@mail.utoronto.ca
Date: 2024-02-15 14:52:45
LastEditors: Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime: 2024-04-01 13:32:18
FilePath: /mbp1413-final/models/unet.py
Description: U-Net model for medical image segmentation
I Love IU
Copyright (c) 2024 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved. 
'''
import monai
import torch
from monai.data import DataLoader
from .network import Network
from typing import Dict, Any


class unet(Network):
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
        super(unet, self).__init__(cfg, lr, epoch, device, name,
                                   optimizer, use_sche,
                                   tr_loader, val_loader, te_loader)
        self.init_model()
        self.init_params()

    def init_model(self) -> None:
        # Create U-Net model
        self.model = monai.networks.nets.UNet(
            spatial_dims=self.cfg.model.spatial_dims,
            in_channels=self.cfg.model.in_channels,
            out_channels=self.cfg.model.out_channels,
            channels=self.cfg.model.channels,
            strides=self.cfg.model.strides,
            kernel_size=self.cfg.model.kernel_size,
            num_res_units=self.cfg.model.num_res_units,
            act=self.cfg.model.activation,
            norm=self.cfg.model.normalization,
            dropout=self.cfg.model.dropout,
        ).to(self.device)
