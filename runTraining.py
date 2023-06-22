import pytorch_lightning as pl
import torch

from model import UNet
from loadData import ImageToImageDataset, shotRandomNoise

model = UNet()
trainer = pl.Trainer(max_epochs=model.hyperparameters['max_epochs'])

