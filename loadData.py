import torch
import torchvision

import numpy as np

from torchvision import transforms
from torch.utils.data import Dataset, Subset, DataLoader

from itertools import permutations




class NoisyCIFAR(object):
    """ object containing small training and validation subsets of CIFAR """
    def __init__(self, trainProfile, valProfile, noise_rate, **_):

        Nimages_train, Mnoisy_train = trainProfile
        Nimages_val, Mnoisy_val = valProfile

        CIFAR10_data = torchvision.datasets.CIFAR10('./CIFAR10', train=True, 
            transform=transforms.ToTensor())

        CIFAR10_test = torchvision.datasets.CIFAR10('./CIFAR10', train=False, transform=transforms.ToTensor())

        imageInds = np.random.choice( range( len(CIFAR10_data) ), 
                        size=(Nimages_train+Nimages_val), replace=False )
        trainInds = torch.tensor(imageInds[0:Nimages_train]).repeat( Mnoisy_train, 1).permute(1,0).reshape( -1 )
        valInds = torch.tensor(imageInds[Nimages_train:]).repeat( Mnoisy_val, 1).permute(1,0).reshape( -1 )

        trainImages = Subset(CIFAR10_data, trainInds)
        valImages = Subset(CIFAR10_data, valInds)

        self.trainInds = trainInds
        self.train_base = next(iter( DataLoader(trainImages, batch_size=len(trainImages)) ))[0]
        self.train_noisy = shotRandomNoise(noise_rate, self.train_base)

        self.valInds = valInds
        self.val_base = next(iter( DataLoader(valImages, batch_size=len(valImages)) ))[0]
        self.val_noisy = shotRandomNoise(noise_rate, self.val_base)

        self.test_base = next( iter( DataLoader(CIFAR10_test, batch_size=500) ))[0]
        self.test_noisy = shotRandomNoise(noise_rate, self.test_base)




# training approaches
class GroundTruthDataset(Dataset):
    """ Dataset of corrupted images and their corresponding ground truth """
    def __init__(self, baseImages, noisyImages):
        super(GroundTruthDataset, self).__init__()
        self.noiseFreeImages = baseImages
        self.noisyImages = noisyImages

    def __len__(self):
        return len(self.noisyImages)

    def __getitem__(self, idx):
        return self.noisyImages[idx], self.noiseFreeImages[idx]



class NoisyNoisyDataset(Dataset):
    """Dataset of pairs (or more) of corrupted imagese"""
    def __init__(self, noisyImages, Nimages, Mnoisy, Kpairs):
        super(NoisyNoisyDataset, self).__init__()
        self.noisyImages = noisyImages

        allPairs = list( permutations( range(Mnoisy), 2 ) )
        pairInds = np.random.choice( range(len(allPairs)), Kpairs, replace=False )

        self.IndexPairs = np.array(allPairs)[pairInds]
        self.ImageIndices = np.concatenate( Nimages*[self.IndexPairs] ) + \
             np.stack( 2*[Mnoisy*np.array(range(Nimages)).repeat((Kpairs))] ).T
        print(np.stack( 2*[Mnoisy*np.array(range(Nimages)).repeat((Kpairs))] ))

    def __len__(self):
        return len(self.noisyImages)

    def __getitem__(self, idx):
        inds = self.ImageIndices[idx]
        return self.noisyImages[inds[0]], self.noisyImages[inds[1]]



## Noise production functions
def shotRandomNoise( rate, images ):
    """ totally randomizes random pixels """
    toNoise = torch.zeros( images.shape )
    toNoise.copy_(images)

    pixelDims = [ images.shape[i] for i in [0,2,3] ]
    noisyPixels = torch.bernoulli( rate * torch.ones( pixelDims ))

    toNoise.permute([0,2,3,1])[ noisyPixels == 1 ] = torch.rand( int(noisyPixels.sum().item()), 3 )

    return toNoise

