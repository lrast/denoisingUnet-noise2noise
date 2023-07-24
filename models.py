import torch

import torch.nn as nn
import pytorch_lightning as pl

from torch.nn.functional import max_pool2d
from torch.utils.data import DataLoader
from .loadData import GroundTruthDataset, NoisyNoisyDataset, NoisyCIFAR


class UNet(pl.LightningModule):
    def __init__(self, **hyperparameters):
        super(UNet, self).__init__()

        # initialize hyperparameters
        hyperparameterValues = {
            'lr': 1e-3,
            'batch_size': 10,

            'dataConstructor': 'groundTruth',
            'Nimages': 10,
            'Mnoisy': 10,
            'Kpairs': 10,
            'noise_rate': 0.1
        }

        hyperparameterValues.update(hyperparameters)
        self.save_hyperparameters(hyperparameterValues)

        # initialize the banks of convolutional filters
        self.desc32 = convBank(3, 6)
        self.desc16 = convBank(6, 12)
        self.desc8 = convBank(12, 24)
        self.desc4 = convBank(24, 48)

        self.middle = convBank(48, 96)

        self.upConv4 = nn.ConvTranspose2d(96, 48, (2, 2), stride=2)
        self.asc4 = convBank(96, 48)
        self.upConv8 = nn.ConvTranspose2d(48, 24, (2, 2), stride=2)
        self.asc8 = convBank(48, 24)
        self.upConv16 = nn.ConvTranspose2d(24, 12, (2, 2), stride=2)
        self.asc16 = convBank(24, 12)
        self.upConv32 = nn.ConvTranspose2d(12, 6, (2, 2), stride=2)
        self.asc32 = convBank(12, 6)

        self.toOut = nn.Conv2d(6, 3, (1, 1), padding='same')

        self.outputNlin = nn.Hardsigmoid()

        # set up loss function
        self.pixelLoss = nn.L1Loss()

    def forward(self, inputs):
        down1 = self.desc32(inputs)
        down2 = self.desc16(max_pool2d(down1, 2))
        down3 = self.desc8(max_pool2d(down2, 2))
        down4 = self.desc4(max_pool2d(down3, 2))
        ubottom = self.middle(max_pool2d(down4, 2))
        up4 = self.asc4(torch.cat((down4, self.upConv4(ubottom)), 1))
        up3 = self.asc8(torch.cat((down3, self.upConv8(up4)), 1))
        up2 = self.asc16(torch.cat((down2, self.upConv16(up3)), 1))
        up1 = self.asc32(torch.cat((down1, self.upConv32(up2)), 1))
        out = self.outputNlin(self.toOut(up1))
        return out

    def training_step(self, batch, batch_ind):
        images, targets = batch
        reconstruction = self.forward(images)

        loss = self.pixelLoss(reconstruction, targets)
        self.log('train_loss', loss.detach())
        return loss

    def validation_step(self, batch, batchidx):
        images, targets = batch
        reconstruction = self.forward(images)
        loss = self.pixelLoss(reconstruction, targets)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    # data
    def setup(self, stage=None, useData=None):
        if useData is None:
            self.data = NoisyCIFAR(
                (self.hparams.Nimages, self.hparams.Mnoisy),
                (500, 1), self.hparams.noise_rate)
        else:
            self.data = useData

    def train_dataloader(self):
        if self.hparams.dataConstructor == 'groundTruth':
            return DataLoader(
                    GroundTruthDataset(
                        self.data.train_base, 
                        self.data.train_noisy),
                    batch_size=self.hparams.batch_size, shuffle=True
                )

        elif self.hparams.dataConstructor == 'noiseToNoise':
            return DataLoader(
                    NoisyNoisyDataset(
                        self.data.train_noisy, 
                        self.hparams.Nimages, self.hparams.Mnoisy, 
                        self.hparams.Kpairs),
                    batch_size=self.hparams.batch_size, shuffle=True
                )

    def val_dataloader(self):
        return DataLoader(
                GroundTruthDataset(self.data.val_base, self.data.val_noisy),
                batch_size=500
                )

    def test_dataloader(self):
        return DataLoader(
                GroundTruthDataset(self.data.test_base, self.data.test_noisy),
                batch_size=self.data.test_base.shape[0]
                )


def convBank(inChannels, outChannels, midChannels=None):
    if midChannels is None:
        midChannels = outChannels

    return nn.Sequential(
            nn.Conv2d(inChannels, midChannels, (3, 3), padding='same'),  # 32
            nn.ReLU(),
            nn.Conv2d(midChannels, outChannels, (3, 3), padding='same'),  # 32
        )


class Control_Mode(object):
    """ Control for data analysis: simply take the mode of the datapoints """
    def __init__(self):
        pass

    def reconstruct(self, noisySamples):
        deNoised, _ = torch.mode(noisySamples, dim=1)
        return deNoised


class ImageSequenceTransformer(pl.LightningModule):
    """Transformers taking a sequence of iamges as input """
    def __init__(self):
        super(ImageSequenceTransformer, self).__init__()
        
        self.network = nn.Sequential(
                nn.Linear(3*32*32, 512),
                nn.Transformer(d_model=512, batch_first=True)
            )

        self.loss = nn.L1Loss()

        self.save_hyperparameters({'lr': 1e-3})

    def training_step(self, batch, batch_ind):
        inputs, targets = batch
        reconstructions = self.forward(inputs)
        return self.loss(targets, reconstructions)

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], 3*32*32)
        output = self.network.forward(x, torch.zeros(x.shape[0], 1, 32*32*3))
        return output.view(x.shape[0], 3, 32, 32)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer



