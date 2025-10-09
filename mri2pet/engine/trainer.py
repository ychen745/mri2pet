import os
import torch
from torch.utils.data import DataLoader
from mri2pet.engine import validator
from mri2pet.cfg import get_cfg
from mri2pet.data.utils import get_nifti_dataset
from mri2pet.utils import DEFAULT_CFG
from mri2pet.utils.torch_utils import init_seeds

class BaseTrainer:
    """
    The base class for creating trainers.

    Methods:
        train: Execute the trianing process
        validate: Run validation on the test set.
        save_model: Save model training checkpoint.
        get_dataset: Get train and validation datasets.
        setup_model: Load or create model.
        build_optimizer: Build optimizer.

    Examples:
        Initialize a trainer and start training
        >>> trainer = BaseTrainer()
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG):
        """
        Initialize the BaseTrainer class.

        Args:
            cfg (str, optional): Path to the configuration file.

        """
        self.cfg = get_cfg(cfg)
        self.device = torch.device(self.cfg['device']) if torch.cuda.is_available() else torch.device('cpu')
        init_seeds(self.cfg['seed'])

        if self.device.type in {"cpu", "mps"}:
            self.cfg['num_workers'] = 0

        self.model = self.get_model()
        self.dataset = self.get_dataset()
        self.train_loader, self.test_loader = self.get_dataloader(self.dataset)
        self.loss_fn = self.get_loss()
        self.validator = self.get_validator()

    def get_model(self):
        """
        To be implemented by the subclass.
        """
        raise NotImplementedError

    def get_dataset(self):
        """
        To be implemented by the subclass.
        Get training, validation and testing datasets from data directory.

        Returns:
            (dict): A dictionary containing training, validation and testing datasets.
        """
        raise NotImplementedError

    def get_dataloader(self, dataset):
        """
        To be implemented by the subclass.
        Get data loader from dataset.
        Returns train and test dataloader.
        """
        raise NotImplementedError

    def get_validator(self):
        """
        To be implemented by the subclass.
        Returns validator object.
        """
        raise NotImplementedError

    def train(self):
        self.model.to(self.device)
        self.model.train()





