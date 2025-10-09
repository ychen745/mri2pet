import os
import torch
from mri2pet.utils.torch_utils import time_sync, intersect_dicts
from .cyclegan import CycleGAN
from collections import OrderedDict
from mri2pet.cfg import get_cfg
# from .pix2pix import Pix2pix
# from .sharegan import ShareGAN

# __all__ = "CycleGAN", "Pix2pix", "ShareGAN"
__all__ = "CycleGAN"

class BaseModel:
    """
    Base class for all generative models.

    This class provides common functionality for generative models including forward pass handling and weight loading capabilities.

    Attributes:
        module_names (List[str]): module names. Used in network printing capabilities.
        training (bool): whether in training mode or not.
        visual_names (List[str]): list of visual names. e.g. real_A, fake_B, ... in GAN
        loss_names (List[str]): list of loss names.
        save_dir (str): directory to save model checkpoints.
        device (torch.device): device to use.

    Methods:
        forward: Perform forward pass for training or inference.
        set_input: set up inputs for the model.
        train: set up training mode.
        eval: set up evaluation mode.
        test: perform forward pass for testing.
        setup: load and print networks
        get_current_visuals: get current visuals.
        get_current_losses: get current losses.
        save_networks: save networks.
        load_networks: load networks.
        print_networks: print networks.
        set_requires_grad: set requires_grad.
        optimize_parameters: optimize network parameters.
    """
    def __init__(self, cfg):
        cfg_dict = get_cfg(cfg)
        self.module_names = [network['name'] for network in cfg_dict['networks']]
        self.training = cfg_dict['training']
        self.visual_names = cfg_dict['visual_names']
        self.loss_names = cfg_dict['loss_names']
        self.save_dir = cfg_dict['checkpoint_path']
        self.device = torch.device(cfg_dict['device']) if 'device' in cfg_dict else 'cpu'

    def forward(self):
        raise NotImplementedError

    def set_input(self, input):
        self.input = input

    # make models eval mode during test time
    def train(self):
        for name in self.module_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.train()

    def eval(self):
        for name in self.module_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.eval()

    def test(self):
        with torch.no_grad():
            self.forward()

    # load and print networks; create schedulers
    def setup(self, cfg):
        # if self.training:
        #     self.schedulers = [networks3D.get_scheduler(optimizer, opt) for optimizer in self.optimizers]

        if not self.training:
            self.load_networks(cfg['which_epoch'], best=cfg['best'])
        elif cfg.continue_train:
            self.load_networks(cfg['which_epoch'], best=False)
        self.print_networks(cfg['verbose'])

    # return visualization images.
    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    # return traning losses/errors.
    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                # float(...) works for both scalar tensor and float number
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    # save models to the disk
    def save_networks(self, which_epoch, best=False):
        for name in self.module_names:
            if isinstance(name, str):
                save_filename = f'best_net_{name}.pth' if best else f'{which_epoch}_net_{name}.pth'
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    # load models from the disk
    def load_networks(self, which_epoch, best=False):
        for name in self.module_names:
            if isinstance(name, str):
                if best:
                    load_filename = f'best_net_{name}.pth'
                else:
                    load_filename = f'{which_epoch}_net_{name}.pth'
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                state_dict = torch.load(load_path, map_location=str(self.device))

                net.load_state_dict(state_dict)

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

    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def optimize_parameters(self):
        raise NotImplementedError
