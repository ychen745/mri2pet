import os
from pathlib import Path
from typing import Dict, Union
from mri2pet.data.augment import Augmentation, RandomCrop
from mri2pet.data.dataset import NiftiDataset

def get_nifti_dataset(cfg, mode='train'):
    """
    Returns training, validation and testing datasets for NiftiDataset.

    Args:
        cfg (dict): kwargs for dataset
        mode (str): 'train' or 'test'

    Returns:
        Dict: Training, validation and testing datasets for NiftiDataset.
    """
    # assert args.data_format == 'nifti'
    transforms = []
    for transform in cfg['transforms']:
        if transform == 'Augmentation':
            transforms.append(Augmentation())
        if transform == 'RandomCrop':
            min_pixel = int(cfg['min_pixel'] * ((cfg['patch_size'][0] * cfg['patch_size'][1] * cfg['patch_size'][2]) / 100))
            transforms.append(RandomCrop(output_size=cfg['patch_size'], drop_ratio=cfg['drop_ratio'], min_pixel=min_pixel))
        else:
            raise NotImplementedError

    return NiftiDataset(cfg['data_path'], transforms, mode)
