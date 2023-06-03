import torch
import loadData

import torch.nn as nn
import pytorch_lightning as pl

from torch.nn.functional import max_pool2d


class UNet(pl.LightningModule):
    def __init__(self, **hyperparameters):
        super(UNet, self).__init__()

        # initialize the banks of convolutional filters
        self.desc32 = convBank(3, 6)
        self.desc16 = convBank(6, 12)
        self.desc8 = convBank(12, 24)
        self.desc4 = convBank(24, 48)

        self.middle = convBank(48, 96)

        self.upConv4 = nn.ConvTranspose2d(96, 48, (2,2), stride=2)
        self.asc4 = convBank(96, 48)
        self.upConv8 = nn.ConvTranspose2d(48, 24, (2,2), stride=2)
        self.asc8 = convBank(48, 24)
        self.upConv16 = nn.ConvTranspose2d(24, 12, (2,2), stride=2)
        self.asc16 = convBank(24, 12)
        self.upConv32 = nn.ConvTranspose2d(12, 6, (2,2), stride=2)
        self.asc32 = convBank(12, 6)

        self.toOut = nn.Conv2d(6,3, (1,1), padding='same')

        # initializer hyperparameters
        self.hyperparameters = {
            'lr': 1e-3,
            'batch_size': 10,

            'Nimages': 10,
            'Mnoisy': 10,
            'noise_rate': 0.1,
            'noise_model': loadData.shotRandomNoise
        }
        self.hyperparameters.update( hyperparameters )

        self.pixelLoss = nn.MSELoss()


    def forward(self, inputs):
        down1 = self.desc32(inputs)
        down2 = self.desc16( max_pool2d(down1, 2) )
        down3 = self.desc8( max_pool2d(down2, 2) )
        down4 = self.desc4( max_pool2d(down3, 2) )
        ubottom = self.middle( max_pool2d(down4, 2) )
        up4 = self.asc4( torch.cat( (down4, self.upConv4(ubottom)), 1) )
        up3 = self.asc8( torch.cat( (down3, self.upConv8(up4)), 1) )
        up2 = self.asc16( torch.cat( (down2, self.upConv16(up3)), 1) )
        up1 = self.asc32( torch.cat( (down1, self.upConv32(up2)), 1) )
        out = self.toOut( up1 )
        return out

    def training_step(self, batch, batch_ind):
        images, targets = batch
        reconstruction = self.forward(images)

        return self.pixelLoss(targets, reconstruction)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hyperparameters['lr'])
        return optimizer

    # data
    def setup( self, stage=None):
        self.trainData = loadData.ImageToImageDataset(**self.hyperparameters)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.trainData, batch_size=self.hyperparameters['batch_size'])




def convBank(inChannels, outChannels, midChannels=None):
    if midChannels is None:
        midChannels = outChannels

    return nn.Sequential(
            nn.Conv2d(inChannels, midChannels, (3,3), padding='same'), #32
            nn.ReLU(),
            nn.Conv2d(midChannels, outChannels, (3,3), padding='same'), #32
        )

