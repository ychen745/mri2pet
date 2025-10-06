import numpy as np
import torch
import torch.nn as nn
from .conv import Conv, Conv3d

class ResnetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, k=3, s=1, padding=0, padding_mode='reflect'):
        """
        Initialize ResNet block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            padding (int, optional): Padding.
            padding_mode (str, optional): Padding mode.
        """
        super().__init__()
        conv_block = []
        p = 0
        if self.padding_mode == 'reflect':
            conv_block += [nn.ReflectionPad3d(1)]
        elif self.padding_mode == 'replicate':
            conv_block += [nn.ReplicationPad3d(1)]
        elif self.padding_mode == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_mode)

        conv_block += Conv3d(c1, c2, k, s, padding=p)
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetBlock3d(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, k=3, s=1, padding=0, padding_mode='reflect'):
        """
        Initialize ResNet block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
        """
        super().__init__()
        conv_block = []
        p = 0
        if self.padding_mode == 'reflect':
            conv_block += [nn.ReflectionPad3d(1)]
        elif self.padding_mode == 'replicate':
            conv_block += [nn.ReplicationPad3d(1)]
        elif self.padding_mode == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_mode)

        conv_block += Conv3d(c1, c2, k, s, padding=p)
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
