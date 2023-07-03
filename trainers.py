import torch
import wandb
import shutil

from models import UNet

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


def runBasicTraining(model, targetdir, group=None, max_epochs=1000):
    """Simple training behavior with checkpointing"""

    wandb.init(project='noiseRemoval', group=group)
    wandb_logger = WandbLogger(project='noiseRemoval', save_dir='wandb/'+targetdir)
    wandb_logger.experiment.config.update(model.hparams )

    checkpoint_callback = ModelCheckpoint(dirpath='lightning_logs/'+targetdir, 
        every_n_epochs=8, save_top_k=4, monitor='val_loss')

    trainer = Trainer(logger=wandb_logger,
        log_every_n_steps=1,
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback])
    trainer.fit(model)

    wandb.finish()
    shutil.copyfile(checkpoint_callback.best_model_path, 'lightning_logs/'+targetdir+'/best.ckpt')


def earlyStoppingTraining(model, targetdir, group=None, patience=100):
    """Training with early stopping"""

    wandb.init(project='noiseRemoval', group=group)
    wandb_logger = WandbLogger(project='noiseRemoval', save_dir='wandb/'+targetdir)
    wandb_logger.experiment.config.update(model.hparams )

    checkpoint_callback = ModelCheckpoint(dirpath='lightning_logs/'+targetdir, 
        every_n_epochs=1, save_top_k=1, monitor='val_loss')
    earlystop_callback = EarlyStopping(monitor='val_loss', mode='min', patience=patience)

    trainer = Trainer(logger=wandb_logger,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback, earlystop_callback])
    trainer.fit(model)

    wandb.finish()


def lossSweep_noisyFree():
    for i in range(9):
        name = 'noiseFree/'
        model = UNet(noise_rate=0.0)
        if i % 3 == 0:
            model.pixelLoss = torch.nn.MSELoss()
            name = name + 'MSE' + str( int( i/3 ))
        elif i % 3 == 1:
            model.pixelLoss = torch.nn.L1Loss()
            name = name + 'L1' + str( int( i/3 ))
        elif i % 3 == 2:
            model.pixelLoss = torch.nn.BCELoss()
            name = name + 'BCE' + str( int( i/3 ))

        earlyStoppingTraining(model, name, group='noiseFree')


def numbersSweep():
    Nimages_vals = [10, 20, 40, 60, 100]
    Mnoisy_vals = [10, 20, 40, 60, 100]

    for Nimages in Nimages_vals:
        for Mnoisy in Mnoisy_vals:
            print(Nimages, Mnoisy)
            model = UNet(Nimages=Nimages, Mnoisy=Mnoisy)
            runBasicTraining(model, 'initialCalibration/'+dataSource+'n'+str(noiseRate) )




