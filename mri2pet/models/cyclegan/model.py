import yaml
import torch
import importlib
from mri2pet.cfg import get_cfg
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from mri2pet.engine.model import Model
from mri2pet.models import BaseModel
from mri2pet.utils.loss import GANLoss

class CycleGAN(Model):
    def __init__(self, model: Union[str, Path] = "cyclegan.pt"):
        """
        Initialize a CycleGAN model.

        This constructor initializes a CycleGAN model.

        Args:
            model (str | Path): Model name or path to model file, i.e. 'cyclegan.pt'.

        Examples:
            >>> from mri2pet import CycleGAN
            >>> model = CycleGAN("cyclegan.pt")  # load a pretrained CycleGAN model
        """
        path = Path(model if isinstance(model, (str, Path)) else "")
        super().__init__(model=model)


class CycleGANModel(BaseModel):
    def __int__(self, cfg=None):
        super().__init__()

        self.cfg = get_cfg(cfg)

        self.device = torch.device(self.cfg["device"]) if torch.cuda.is_available() else torch.device('cpu')

        self.define_networks()

        # Visuals
        self.real_A = None
        self.real_B = None
        self.fake_A = None
        self.fake_B = None

        # Loss functions
        self.criterionGAN = GANLoss(use_lsgan=False).to(self.device)
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionIdt = torch.nn.L1Loss()

        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.G1.parameters(), lr=self.cfg["lr"], betas=(self.cfg["beta1"], 0.999))
        self.optimizer_D = torch.optim.Adam(self.D1.parameters(), lr=self.cfg["lr"], betas=(self.cfg["beta1"], 0.999))

    @staticmethod
    def get_cfg(cfg):
        with open(cfg) as f:
            cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        return cfg_dict

    def set_input(self, input):
        self.real_A = input[0].to(self.device)
        self.real_B = input[1].to(self.device)

    def define_networks(self):
        modules = self.cfg.get("module_names")
        for module in modules:
            setattr(self, module.name, self.create_module(module.module_name))

    @staticmethod
    def create_module(module_name: str):
        return getattr(importlib.import_module("mri2pet.nets"), module_name)

    def forward(self):
        self.fake_B = self.G1(self.real_A)
        self.rec_A = self.G2(self.fake_B)
        self.fake_A = self.G2(self.real_B)
        self.rec_B = self.G1(self.fake_A)

    def loss(self, batch, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | List[torch.Tensor], optional): Predictions.
        """

        preds = self.forward(batch["img"]) if preds is None else preds
        return self.criterion(preds, batch)

    def backward_G(self):
        lambda_idt = self.cfg['lambda_identity']
        lambda_A = self.cfg['lambda_A']
        lambda_B = self.cfg['lambda_B']

        # Identity loss
        if lambda_idt > 0:
            self.idt_A = self.netG1(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            self.idt_B = self.netG2(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss
        self.loss_G1 = self.criterionGAN(self.netD1(self.fake_B), True)
        self.loss_G2 = self.criterionGAN(self.netD2(self.fake_A), True)
        # Cycle loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss
        self.loss_G = self.loss_G1 + self.loss_G2 + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def backward_D(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D1(self):
        # fake_B = self.fake_B_pool.query(self.fake_B)
        # self.loss_D1 = self.backward_D(self.netD_A, self.real_B, fake_B)
        self.loss_D1 = self.backward_D(self.netD1, self.real_B, self.fake_B)

    def backward_D2(self):
        # fake_A = self.fake_A_pool.query(self.fake_A)
        # self.loss_D2 = self.backward_D(self.netD_B, self.real_A, fake_A)
        self.loss_D2 = self.backward_D(self.netD2, self.real_A, self.fake_A)

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.set_requires_grad([self.netD1, self.netD2], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A and D_B
        self.set_requires_grad([self.netD1, self.netD2], True)
        self.optimizer_D.zero_grad()
        self.backward_D1()
        self.backward_D2()
        self.optimizer_D.step()

