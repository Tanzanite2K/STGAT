import torch

def MAE(pred, true):
    return torch.mean(torch.abs(pred - true)).item()

def RMSE(pred, true):
    return torch.sqrt(torch.mean((pred - true) ** 2)).item()
