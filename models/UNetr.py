"""
Author: Chris Xiao yl.xiao@mail.utoronto.ca
Date: 2024-04-02 00:37:26
LastEditors: Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime: 2024-04-02 00:40:52
FilePath: /mbp1413-final/models/unetr.py
Description: UNetr model
I Love IU
Copyright (c) 2024 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved.
"""

__all__ = ["unetr"]
import monai
import torch
from models.network import Network
from typing import Dict, Any
from monai.data import DataLoader


class unetr(Network):
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
        te_loader: DataLoader,
    ) -> None:
        super(unetr, self).__init__(
            cfg,
            lr,
            epoch,
            device,
            name,
            optimizer,
            use_sche,
            tr_loader,
            val_loader,
            te_loader,
        )
        self.init_model()
        self.init_params()

    def init_model(self) -> None:
        # Create U-NetR model
        self.model = monai.networks.nets.UNETR(
            spatial_dims=self.cfg.model.spatial_dims,
            in_channels=self.cfg.model.in_channels,
            out_channels=self.cfg.model.out_channels,
            img_size=self.cfg.model.image_size,
            norm_name=self.cfg.model.normalization,
            dropout_rate=self.cfg.model.dropout,
        ).to(self.device)
