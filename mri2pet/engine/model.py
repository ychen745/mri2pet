import os
import torch
import yaml
import importlib
from collections import OrderedDict
from mri2pet.utils import utils

class Model(torch.nn.Module):
    """
    A base class for implementing generative models.

    Attributes:
        model (torch.nn.Module): The underlying PyTorch model.
        trainer (BaseTrainer): The trainer object used for training the model.
        predictor (BasePredictor): The predictor object used for making predictions.
        cfg (str): The configuration of the model if loaded from a *.yaml file.
        ckpt_path (str): The path to the checkpoint file.

    Methods:
        __call__: Alias for the predict method, enabling the model instance to be callable.
        create_model: Initialize a new model based on a configuration file.
        load_weights: Load model weights from a checkpoint file.
        save_weights: Save the current state of the model to a file.
        predict: Perform predictions.
        train: Train the model on a dataset.
        val: Validate the model on a dataset.
    """
    def __init__(self, cfg='CycleGAN.yaml', weights=None):
        super().__init__()

        self.model = None
        self.trainer = None
        self.predictor = None
        self.ckpt_path = None
        self.cfg = cfg

        with open(cfg) as f:
            cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        self.cfg_dict = cfg_dict

        self.create_model()
        if weights is not None:
            self.ckpt_path = weights
            self.load_weights(weights)

    def forward(self):
        pass

    def __call__(self, **kwargs):
        """
        Alias for the predict method, enabling the model instance to be callable for predictions.
        """
        return self.predict(**kwargs)

    @staticmethod
    def create_module(module_name: str):
        return getattr(importlib.import_module(f"mri2pet.nets.{module_name}"), module_name)

    def load_weights(self, weights: str):
        """
        Load a model from a checkpoint file.
        """
        try:
            for name in self.model.keys():
                state_dict = torch.load(weights, weights_only=True)
                self.model[name].load_state_dict(state_dict)
        except:
            print(f'Failed to load model weights.')

    def save_weights(self, weights: str):
        """
        Save a model to a checkpoint file.
        """
        try:
            for name in self.model.keys():
                torch.save(self.model[name].state_dict(), weights)
        except:
            print(f'Failed to save model weights.')

    # create schedulers and load weights for training
    def setup(self, opt):
        if self.isTrain:
            self.schedulers = [utils.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain:
            self.load_networks(opt.which_epoch, best=opt.best)
        elif opt.continue_train:
            self.load_networks(opt.which_epoch)
        self.print_networks(verbose=True)


    def train(self, trainer=None, **kwargs):
        """
        Train the model using the specified dataset and training configuration.

        This method facilitates model training with a range of customizable settings. It supports training with a
        custom trainer or the default training approach.

        Args:
            trainer (BaseTrainer, optional): Custom trainer instance for model training. If None, uses default.
            **kwargs (Any): Arbitrary keyword arguments for training configuration. Common options include:
                data (str): Path to dataset configuration file.
                epochs (int): Number of training epochs.
                batch (int): Batch size for training.
                imgsz (int): Input image size.
                device (str): Device to run training on (e.g., 'cuda', 'cpu').
                workers (int): Number of worker threads for data loading.
                optimizer (str): Optimizer to use for training.
                lr0 (float): Initial learning rate.
                patience (int): Epochs to wait for no observable improvement for early stopping of training.

        Returns:
            (Dict | None): Training metrics if available and training is successful; otherwise, None.

        Examples:
            >>> model = CycleGAN("cyclegan.pt")
            >>> results = model.train(data="cyclegan.yaml", epochs=3)
        """
        if kwargs.get("cfg"):
            with open(kwargs["cfg"]) as f:
                overrides = yaml.safe_load(kwargs.get("cfg"))
        else:
            overrides = self.overrides

        self.trainer = trainer or self.setup_trainer()
        self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
        self.model = self.trainer.model
        self.trainer.train()

    # used in test time, wrapping `forward` in no_grad() so we don't save
    # intermediate steps for backprop
    def test(self):
        with torch.no_grad():
            self.forward()

    def setup_trainer(self):
        pass

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    # return visualization images. train.py will display these images, and save the images to a html
    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    # return traning losses/errors. train.py will print out these errors as debugging information
    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                # float(...) works for both scalar tensor and float number
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    # save models to the disk
    def save_networks(self, which_epoch, best=False):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = 'best_net_%s.pth' % (name) if best else '%s_net_%s.pth' % (which_epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    # load models from the disk
    def load_networks(self, which_epoch, best=False):
        for name in self.model_names:
            if best:
                load_filename = 'best_net_%s.pth' % (name)
            else:
                load_filename = '%s_net_%s.pth' % (which_epoch, name)

    # print network information
    def print_networks(self, verbose=False):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    # set requies_grad=Fasle to avoid computation
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
