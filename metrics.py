import torch


def MSLoss( reconstruction, targets ):
    l = torch.nn.MSELoss()
    return l(reconstruction, targets).item()


def AvgIncorrectPixels(reconstruction, targets):
    return (reconstruction != targets).sum( dim=[1,2,3]).float().mean().item()
