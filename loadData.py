import torch
import torchvision

import numpy as np

from torchvision import transforms
from torch.utils.data import Dataset


class ImageToImageDataset(Dataset):
    def __init__(self, Nimages, Mnoisy, makeNoise=None):
        super(ImageToImageDataset, self).__init__()

        CIFAR10_data = torchvision.datasets.CIFAR10('./CIFAR10', train=True, download=False, 
            transform=transforms.ToTensor())
        imageInds = np.random.choice( range( len(CIFAR10_data) ), size=Nimages, replace=False )


        # generate the noisy image dataset
        self.inputs = []
        self.targets = []

        for index in imageInds:
            image, label = CIFAR10_data.__getitem__(index)

            for i in range(Mnoisy):
                self.inputs.append( image )
                self.targets.append( shotRandomNoise(0.1, image) )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]



## Noise production functions
def shotRandomNoise( rate, image ):
    noisyPixels = torch.bernoulli( rate * torch.ones( image.shape[-2:] ))
    noise = (torch.rand( image.shape ) + 1) * noisyPixels 

    return (image + noise) % 1

