import torch
import wandb

from models import UNet

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


def runBasicTraining(model):
    """Simple training behavior with checkpointing and early stopping"""
    max_epochs = 10000 # hopefully it never gets this far

    wandb_logger = WandbLogger(project='noiseRemoval')
    wandb_logger.experiment.config.update(model.hyperparameters )

    checkpoint_callback = ModelCheckpoint(every_n_epochs=8, save_top_k=4, monitor='Train Loss')
    earlystop_callback = EarlyStopping(monitor='val_loss', mode='min', patience=8)

    trainer = Trainer(logger=wandb_logger,
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, earlystop_callback])
    trainer.fit(model)


