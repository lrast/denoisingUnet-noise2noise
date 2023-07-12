import torch


def L1Loss( reconstruction, targets ):
    l = torch.nn.L1Loss()
    return l(reconstruction, targets).item()


def AvgIncorrectPixels(reconstruction, targets):
    return (reconstruction != targets).sum( dim=[1,2,3]).float().mean().item()
