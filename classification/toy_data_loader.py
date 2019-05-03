import os.path as osp
import sys
import torch
import torch.utils.data as data
import numpy as np

class ToyData(data.Dataset):

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels


    def __len__(self):
        return (len(self.images))

    def __getitem__(self, index):

        image = torch.Tensor(self.images[index])
        image = image.permute(2, 0, 1)
        label = torch.Tensor([self.labels[index]])

        return(image, label)
