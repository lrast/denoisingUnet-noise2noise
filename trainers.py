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


def earlyStoppingTraining(model, targetdir, group=None, patience=1000):
    """Training with early stopping"""

    wandb.init(project='noiseRemoval', group=group)
    wandb_logger = WandbLogger(project='noiseRemoval', save_dir='wandb/'+targetdir)
    wandb_logger.experiment.config.update(model.hparams )

    checkpoint_callback = ModelCheckpoint(dirpath='lightning_logs/'+targetdir, 
        every_n_epochs=1, save_top_k=1, monitor='val_loss')
    earlystop_callback = EarlyStopping(monitor='val_loss', mode='min', patience=patience)

    trainer = Trainer(logger=wandb_logger,
        log_every_n_steps=1,
        max_epochs=2500,
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
    """ How does reproduction accuracy scale with number of images? """

    def initAndTrain():
        wandb.init( group='groundTruthSweep')

        N = wandb.config.Nimages
        M = wandb.config.Mnoisy
        nr = wandb.config.noise_rate
        rep = wandb.config.replicate

        model = UNet(**wandb.config)
        targetdir = 'groundTruthSweep/N{N}M{M}nr{nr}rep{rep}'.format(N=N, M=M, nr=nr, rep=rep)

        checkpoint_callback = ModelCheckpoint(dirpath='lightning_logs/'+targetdir, 
            every_n_epochs=1, save_top_k=1, monitor='val_loss')
        earlystop_callback = EarlyStopping(monitor='val_loss', mode='min', patience=1000)

        wandb_logger = WandbLogger(project='noiseRemoval', save_dir='wandb/'+targetdir)

        trainer = Trainer(logger=wandb_logger,
            log_every_n_steps=1,
            max_epochs=10000,
            callbacks=[checkpoint_callback, earlystop_callback])
        trainer.fit(model)

        wandb.finish()


    sweep_configuration = {
        'method': 'grid',
        'name': 'Number of Samples',
        'parameters': 
        {
            'Nimages': {'values': [10, 20, 50, 100]},
            'Mnoisy': {'values': [10, 20, 50, 100]},
            'noise_rate': {'values': [0, 0.2, 0.4, 0.6, 0.8, 0.9]},
            'replicate': {'values': [0,1,2]}
         }
    }

    sweep_id = wandb.sweep(sweep_configuration, project='noiseRemoval')
    wandb.agent(sweep_id, function=initAndTrain)


