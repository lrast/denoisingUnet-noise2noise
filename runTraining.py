import pytorch_lightning as pl
import torch
from model import UNet
from loadData import ImageToImageDataset, shotRandomNoise

model = UNet()
imageData = ImageToImageDataset(10, 10, shotRandomNoise)
dl = torch.utils.data.DataLoader( imageData, batch_size=10)


t = pl.Trainer()

