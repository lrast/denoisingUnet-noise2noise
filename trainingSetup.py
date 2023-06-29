import pytorch_lightning as pl
import torch

from models import UNet
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


model = UNet()
trainer = pl.Trainer(max_epochs=model.hyperparameters['max_epochs'], 
    callbacks=[EarlyStopping(monitor='val_loss', mode='min', patience=8)] )



