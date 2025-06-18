import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

def get_optimizer(model, lr: float = 1e-4):
    return Adam(model.parameters(), lr=lr)

def get_scheduler(optimizer, patience: int = 3, factor: float = 0.5):
    return ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=factor)
