import os
from pathlib import Path
from typing import Dict, Union
from mri2pet.data.augment import Augmentation, RandomCrop
from mri2pet.data.dataset import NiftiDataset

def get_nifti_dataset(args, mode='train'):
    """
    Returns training, validation and testing datasets for NiftiDataset.

    Args:
        args (dict): kwargs for dataset

    Returns:
        Dict: Training, validation and testing datasets for NiftiDataset.
    """
    assert args.data_format == 'nifti'
    transforms = []
    for transform in args.transforms:
        if transform == 'Augmentation':
            transforms.append(Augmentation())
        if transform == 'RandomCrop':
            min_pixel = int(args.min_pixel * ((args.patch_size[0] * args.patch_size[1] * args.patch_size[2]) / 100))
            transforms.append(RandomCrop(output_size=args.patch_size, drop_ratio=args.drop_ratio, min_pixel=min_pixel))
        else:
            raise NotImplementedError

    return NiftiDataset(args.data_path, transforms, mode)
