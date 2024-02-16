'''
Author: Chris Xiao yl.xiao@mail.utoronto.ca
Date: 2024-02-15 14:52:45
LastEditors: Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime: 2024-02-15 22:55:53
FilePath: /mbp1413-final/models/unet.py
Description: U-Net model for medical image segmentation
I Love IU
Copyright (c) 2024 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved. 
'''
from typing import Mapping
import monai
import torch
import torch.nn as nn
from .network import Network
from typing import Dict, Any


class unet(Network):
    def __init__(
        self,
        cfg: Dict[str, Any],
        device: torch.device,
        resume: bool = False
    ) -> None:
        super(unet, self).__init__(cfg, device)
        self.init_model()
        self.init_params()
        if resume:
            self.load_checkpoint()
    
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
    