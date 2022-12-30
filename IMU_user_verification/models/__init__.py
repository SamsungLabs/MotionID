from .layers import Cat, SupConLoss
from .os_net import Net as OsNet

from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

model_by_name_dict = {
    'os_net': OsNet,
    'none': OsNet,
}


optimizer_by_name_dict = {
    'sgd': SGD,
    'adam': Adam,
    'none': SGD,
}

lr_scheduler_by_name_dict = {
    'cosine_annealing': CosineAnnealingLR,
    'step': StepLR,
    'none': StepLR
}


def model_by_name(name='os_net'):
    if name is None or name not in model_by_name_dict:
        return model_by_name_dict['none']
    else:
        return model_by_name_dict[name]


def optimizer_by_name(name='sgd'):
    if name is None or name not in optimizer_by_name_dict:
        return optimizer_by_name_dict['none']
    else:
        return optimizer_by_name_dict[name]


def lr_scheduler_by_name(name='sgd'):
    if name is None or name not in lr_scheduler_by_name_dict:
        return lr_scheduler_by_name_dict['none']
    else:
        return lr_scheduler_by_name_dict[name]


__all__ = ['Cat', 'SupConLoss', 'OsNet', 'model_by_name', 'optimizer_by_name', 'lr_scheduler_by_name']

