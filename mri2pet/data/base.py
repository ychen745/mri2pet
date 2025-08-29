import os
import torch

class BaseDataset(torch.utils.data.Dataset):
    """
    Base Dataset class.
    """
    def __init__(self, data_path, transforms=None, mode='train'):
        self.data_path = data_path
        self.transforms = transforms
        self.mode = mode
        self.images = []
        self.labels = []

        if self.mode == 'train':
            fdata = os.path.join(data_path, 'train.txt')
        else:
            fdata = os.path.join(data_path, 'test.txt')

        with open(fdata, 'r') as f:
            samples = f.read().splitlines()
            if samples[0] == 'images,labels':
                samples = samples[1:]
            self.images, self.labels = map(lambda x: x.strip().split(','), samples)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]



