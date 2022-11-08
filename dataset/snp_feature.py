import os
import torch.utils.data as data
import numpy as np
import random
from torchvision.transforms import ToTensor
from torchvision import transforms

class SNPFeature(data.Dataset):
    """input and label image dataset"""

    def __init__(self, X, GT1, GT2, classdict={}, transform=False):
        super(SNPFeature, self).__init__()
        """
        Args:
        X: inputs, SNP features, each feature is a binary code.
        GT1: haplotype labels
        GT2: haplotype labels
        transform(callable, optional): Optional transform to be applied on a sample
        """
        self.X = X
        self.GT1 = GT1
        self.GT2 = GT2
        self.transform = transform

    def __getitem__(self, index):
        sample = {}
        sample['SNP'] = self.X
        sample['hap1'] = self.GT1
        sample['hap2'] = self.GT2

        return sample

    def __len__(self):
        return len(self.ids)
