import math
import random
from copy import copy
from typing import Any, Dict, List, Optional
import torch.nn as nn
from torch.utils.data import DataLoader
from mri2pet.data.utils import get_nifti_dataset
from mri2pet.engine.trainer import BaseTrainer
from .model import CycleGANModel


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
    """

    def __init__(self, cfg: str = "CycleGAN.yaml"):
        super().__init__(cfg)

        self.model = self.get_model()
        self.dataset = self.get_dataset()
        self.train_loader, self.test_loader = self.get_dataloader(self.dataset)
        self.loss = self.get_loss(cfg)
        self.validator = self.get_validator(cfg)

    def get_model(self):
        model = CycleGANModel(self.cfg)
        return model

    def get_dataset(self):
        train_dataset = get_nifti_dataset(self.cfg, 'train')
        test_dataset = get_nifti_dataset(self.cfg, 'test')
        return {'train': train_dataset, 'test': test_dataset}

    def get_dataloader(self, dataset):
        train_loader = DataLoader(self.dataset['train'], num_workers=self.cfg['num_workers'], batch_size=self.cfg['batch_size'], shuffle=True)
        test_loader = DataLoader(self.dataset['test'], num_workers=0, batch_size=1, shuffle=True)
        return train_loader, test_loader

    def get_validator(self):
        """Return a DetectionValidator for YOLO model validation."""
        return None

    def train(self):
        pass