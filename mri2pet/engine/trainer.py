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

    This class provides foundations for training generative models, handling the training loop, validation, checkpointing,
    and various training utilities. It supports both single-GPU and multi-GPU distributed training.

    Attributes:
        args (dict): Configurations for the trainer.
        validator (BaseValidator): Validator instance.
        model (nn.Module): Model instance.
        res_dir (Path): Directory to save results.
        checkpoint_dir (Path): Directory to save checkpoints.
        last (Path): Directory to save last checkpoint.
        best (Path): Directory to save the best checkpoint.
        save_period (int): Save checkpoint every x epochs (disabled if < 1).
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
        start_epoch (int): Starting epoch.
        device (torch.device): Device used for training.
        data (str): Path to the training data.
        resume (bool): Whether to resume training from a checkpoint.
        loss_function (nn.Module): Loss function.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        loss (float): Current loss value.
        total_loss (float): Total loss value.
        csv (Path): Path to the results CSV file.
        metrics (dict): Metrics dictionary.
        plots (dict): Plots dictionary.

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
        self.args = get_cfg(cfg)
        self.device = torch.device(self.args.device) if torch.cuda.is_available() else torch.device('cpu')
        self.validator = None
        self.metrics = None
        self.plot = {}
        init_seeds(self.args.seed)

        self.res_dir = self.args.res_dir
        self.checkpoint_dir = self.args.checkpoint_dir
        self.last, self.best = self.checkpoint_dir / "last.pt", self.checkpoint_dir / "best.pt"
        self.save_period = self.args.save_period

        self.batch_size = self.args.batch_size
        self.epochs = self.args.epochs
        self.start_epoch = self.args.start_epoch

        if self.device.type in {"cpu", "mps"}:
            self.args.workers = 0

        self.model = None
        self.data = self.get_dataset()
        self.loss_fn = self.get_loss()

    def get_dataset(self):
        """
        Get training, validation and testing datasets from data directory.

        Returns:
            (dict): A dictionary containing training, validation and testing datasets.
        """
        train_dataset = get_nifti_dataset(self.args, 'train')
        test_dataset = get_nifti_dataset(self.args, 'test')
        return {'train': train_dataset, 'test': test_dataset}

    def get_dataloader(self):
        train_loader = DataLoader(self.data['train'], num_workers=self.args.workers, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(self.data['test'], num_workers=self.args.workers, batch_size=1, shuffle=True)
        return train_loader, test_loader

    def get_loss(self):
        pass

    def train(self):
        train_loader, test_loader = self.get_dataloader()
        self.model.train()
        self.model.to(self.device)



