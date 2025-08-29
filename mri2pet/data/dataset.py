import SimpleITK as sitk
import numpy as np
import random
import torch
import torch.utils.data
from pathlib import Path
from typing import Union
from base import BaseDataset

class NiftiDataset(BaseDataset):
    """
    Dataset class for the Nifti data.

    Args:
        data_path (Union[Path, str]): Data path
        transforms (Transforms): Image transforms
        target_transforms (Transforms): Target transforms
        mode (str): 'train' or 'test'
        shuffle_labels (bool): Shuffle labels for unpaired training

    Methods:
        read_image: Read Nifti image with SimpleITK
    """
    def __init__(self, data_path, transforms=None, mode='train', shuffle_labels=False):
        super().__init__(data_path, transforms, mode)

        self.shuffle_labels = shuffle_labels
        self.bit = sitk.sitkFloat32

    def read_image(self, path):
        reader = sitk.ImageFileReader()
        reader.SetFileName(path)

        try:
            image = reader.Execute()
        except:
            print(f'Failed to read image: {path}')

        return image

    def normalize255(image):
        """
        Normalize an image to 0 - 255 (8bits)

        Args:
            image (sitk Image): Image to normalize

        Returns:
            image (sitk Image): Normalized image
        """
        normalizeFilter = sitk.NormalizeImageFilter()
        resacleFilter = sitk.RescaleIntensityImageFilter()
        resacleFilter.SetOutputMaximum(255)
        resacleFilter.SetOutputMinimum(0)

        image = normalizeFilter.Execute(image)  # set mean and std deviation
        image = resacleFilter.Execute(image)  # set intensity 0-255

        return image

    def __getitem__(self, index):
        image_path = self.images[index]
        label_path = self.labels[index]
        if self.shuffle_labels:
            index_b = random.randint(0, len(self.labels) - 1)
            label_path = self.labels[index_b]

        # read, normalize and cast precision of images
        image = self.read_image(image_path)
        image = self.normalize255(image)
        castImageFilter = sitk.CastImageFilter()
        castImageFilter.SetOutputPixelType(self.bit)
        image = castImageFilter.Execute(image)

        # read, normalize and cast precision of labels
        label = self.read_image(label_path)
        label = self.normalize255(label)  # set intensity 0-255
        castImageFilter.SetOutputPixelType(self.bit)
        label = castImageFilter.Execute(label)

        sample = {'image': image, 'label': label}

        if self.transforms:  # apply the transforms to image and label (normalization, resampling, patches)
            for transform in self.transforms:
                sample = transform(sample)

        # convert sample to tf tensors
        image_np = sitk.GetArrayFromImage(sample['image'])
        label_np = sitk.GetArrayFromImage(sample['label'])

        # to unify matrix dimension order between SimpleITK([x,y,z]) and numpy([z,y,x])  (actually it´s the contrary)
        image_np = np.transpose(image_np, (2, 1, 0))
        label_np = np.transpose(label_np, (2, 1, 0))

        label_np = (label_np - 127.5) / 127.5
        image_np = (image_np - 127.5) / 127.5

        image_np = image_np[np.newaxis, :, :, :]
        label_np = label_np[np.newaxis, :, :, :]

        return torch.from_numpy(image_np), torch.from_numpy(label_np)


