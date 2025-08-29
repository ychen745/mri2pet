import os
import torch
import yaml
from collections import OrderedDict
from mri2pet.models.utils import init_weights, get_scheduler


class Model(torch.nn.Module):
    """
    A base class for implementing generative models, unifying APIs across different model types.

    This class provides a common interface for various operations related to generative models, including training,
    validation and prediction. It handles different types of models.

    Attributes:
        model (torch.nn.Module): The underlying PyTorch model.
        trainer (BaseTrainer): The trainer object used for training the model.
        predictor (BasePredictor): The predictor object used for making predictions.
        ckpt (dict): The checkpoint data if the model is loaded from a *.pt file.
        cfg (str): The configuration of the model if loaded from a *.yaml file.
        ckpt_path (str): The path to the checkpoint file.
        metrics (dict): The latest training/validation metrics.
        model_name (str): The name of the model.

    Methods:
        __call__: Alias for the predict method, enabling the model instance to be callable.
        _new: Initialize a new model based on a configuration file.
        _load: Load a model from a checkpoint file.
        reset_weights: Reset the model's weights to their initial state.
        load: Load model weights from a specified file.
        save: Save the current state of the model to a file.
        info: Log or return information about the model.
        predict: Perform predictions.
        train: Train the model on a dataset.
        val: Validate the model on a dataset.
        tune: Perform hyperparameter tuning.
        _apply: Apply a function to the model's tensors.
    """
    def __init__(self, model='CycleGAN'):
        super().__init__()

        self.model = None
        self.trainer = None
        self.predictor = None
        self.ckpt = {}
        self.cfg = None
        self.ckpt_path = None
        self.metrics = None
        self.model_name = None
        model = str(model).strip()

        if str(model).endswith((".yaml", ".yml")):
            self._new(model)
        else:
            self._load(model)

    def forward(self):
        pass

    def __call__(self, **kwargs):
        """
        Alias for the predict method, enabling the model instance to be callable for predictions.
        """
        return self.predict(**kwargs)

    def _new(self, cfg: str):
        """
        Initialize a new model from model definitions.

        Creates a new model instance based on the provided configuration file. Loads the model configuration
        and initializes the model using the appropriate class.

        Args:
            cfg (str): Path to the model configuration file in YAML format.

        Raises:
            ValueError: If the configuration file is invalid.

        Examples:
            >>> model = Model()
            >>> model._new("cyclegan.yaml")
        """
        with open(cfg) as f:
            cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        self.cfg = cfg


    def _load(self, weights: str, task=None) -> None:
        """
        Load a model from a checkpoint file or initialize it from a weights file.

        This method handles loading models from either .pt checkpoint files or other weight file formats. It sets
        up the model, task, and related attributes based on the loaded weights.

        Args:
            weights (str): Path to the model weights file to be loaded.

        Raises:
            FileNotFoundError: If the specified weights file does not exist or is inaccessible.
            ValueError: If the weights file format is unsupported or invalid.

        Examples:
            >>> model = Model()
            >>> model._load("cyclegan.pt")
            >>> model._load("path/to/weights.pth")
        """
        if weights.lower().startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://")):
            weights = checks.check_file(weights, download_dir=SETTINGS["weights_dir"])  # download and return local file
        weights = checks.check_model_file_from_stem(weights)  # add suffix, i.e. yolo11n -> yolo11n.pt

        if str(weights).rpartition(".")[-1] == "pt":
            self.model, self.ckpt = attempt_load_one_weight(weights)
            self.task = self.model.task
            self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)
            self.ckpt_path = self.model.pt_path
        else:
            weights = checks.check_file(weights)  # runs in all cases, not redundant with above call
            self.model, self.ckpt = weights, None
            self.task = task or guess_model_task(weights)
            self.ckpt_path = weights
        self.overrides["model"] = weights
        self.overrides["task"] = self.task
        self.model_name = weights

    # load and print networks; create schedulers
    def setup(self, opt, parser=None):
        if self.isTrain:
            self.schedulers = [networks3D.get_scheduler(optimizer, opt) for optimizer in self.optimizers]

        if not self.isTrain:
            self.load_networks(opt.which_epoch, best=opt.best)
        elif opt.continue_train:
            self.load_networks(opt.which_epoch)
        self.print_networks(opt.verbose)

    # make models eval mode during test time
    def eval(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def train(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()

    # used in test time, wrapping `forward` in no_grad() so we don't save
    # intermediate steps for backprop
    def test(self):
        with torch.no_grad():
            self.forward()

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def optimize_parameters(self):
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
            if isinstance(name, str):
                if best:
                    load_filename = 'best_net_%s.pth' % (name)
                else:
                    load_filename = '%s_net_%s.pth' % (which_epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))

                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    # print network information
    def print_networks(self, verbose):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
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
