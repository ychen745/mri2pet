# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import math
import random
from copy import copy
from typing import Any, Dict, List, Optional
import numpy as np
import torch.nn as nn
from mri2pet.engine.trainer import BaseTrainer
from mri2pet.models import cyclegan


class CycleGANTrainer(BaseTrainer):
    """
    A class extending the BaseTrainer class for training a CycleGAN model.

    Attributes:
        model (CycleGANModel): The CycleGAN model being trained.
        data (Dict): Dictionary containing dataset information.
        loss (tuple): Names of the loss components used in training (GANLoss, CycleLoss, IdentityLoss, etc.).

    Methods:
        get_model: Return a CycleGAN model.
        get_validator: Return a validator for model evaluation.

    Examples:
        >>> from mri2pet.models.cyclegan import CycleGANTrainer
        >>> args = dict(model="cyclegan.pt", data="cyclegan.yaml", epochs=3)
        >>> trainer = CycleGANTrainer()
        >>> trainer.train()
    """

    def get_model(self, cfg: Optional[str] = None, weights: Optional[str] = None, verbose: bool = True):
        """
        Return a CycleGAN model.

        Args:
            cfg (str, optional): Path to model configuration file.
            weights (str, optional): Path to model weights.

        Returns:
            (CycleGANModel): CycleGAN model.
        """
        model = CycleGANModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )