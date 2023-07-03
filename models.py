import torch
import loadData

import torch.nn as nn
import pytorch_lightning as pl

from torch.nn.functional import max_pool2d
from torch.utils.data import DataLoader


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

        hyperparameterValues.update( hyperparameters )
        self.save_hyperparameters( hyperparameterValues )

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

        self.outputNlin = nn.Hardsigmoid()

        # set up loss function
        self.pixelLoss = nn.L1Loss()
        self.mse = nn.MSELoss()


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
        out = self.outputNlin( self.toOut( up1 ) )
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
    def setup( self, stage=None, useData=None):
        if useData is None:
            self.data = loadData.NoisyCIFAR( 
                (self.hparams.Nimages, self.hparams.Mnoisy), 
                (100, 1), self.hparams.noise_rate)
        else:
            self.data = useData

    def train_dataloader(self):
        if self.hparams.dataConstructor == 'groundTruth':
            return DataLoader( 
                    loadData.GroundTruthDataset(self.data.train_base, self.data.train_noisy),
                    batch_size=self.hparams.batch_size, shuffle=True
                )

        elif self.hparams.dataConstructor == 'noiseToNoise':
            return DataLoader(
                    loadData.NoisyNoisyDataset(self.data.train_noisy, 
                    self.hparams.Nimages, self.hparams.Mnoisy, 
                    self.hparams.Kpairs),
                    batch_size=self.hparams.batch_size, shuffle=True
                )

    def val_dataloader(self):
        return DataLoader( loadData.GroundTruthDataset(self.data.val_base, self.data.val_noisy ),
                    batch_size=self.data.val_base.shape[0] )

    def test_dataloader(self):
        return DataLoader( loadData.GroundTruthDataset(self.data.test_base, self.data.test_noisy ),
                    batch_size=self.data.test_base.shape[0] )



def convBank(inChannels, outChannels, midChannels=None):
    if midChannels is None:
        midChannels = outChannels

    return nn.Sequential(
            nn.Conv2d(inChannels, midChannels, (3,3), padding='same'), #32
            nn.ReLU(),
            nn.Conv2d(midChannels, outChannels, (3,3), padding='same'), #32
        )




class Control_Mode(object):
    """ Control for data analysis: simply take the mode of the datapoints """
    def __init__(self, Mnoisy):
        self.Mnoisy = Mnoisy

    def reconstruct(self, sortedDataset):
        reconstructions = []
        groundTruths = []

        for startInd in range(0, len(sortedDataset), self.Mnoisy):
            inputs, targets = sortedDataset[ startInd : (startInd+self.Mnoisy) ]

            allInputs = torch.stack( inputs )
            deNoised, _ = torch.mode(allInputs, dim=0)

            reconstructions.append( deNoised )
            groundTruths.append( targets[0] )

        return torch.stack(reconstructions), torch.stack(groundTruths)



