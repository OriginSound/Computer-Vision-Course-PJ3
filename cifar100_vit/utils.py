import os, sys 
from typing import Any, Callable, Optional, Tuple
import torch.nn as nn

from models import * 

def get_network(name):
    name = name.lower()
    if name == "vit_small":
        net = ViT_Small()
    elif name == 'resnet18':
        net = ResNet18()
    elif name == 'resnet34':
        net = ResNet34()
    elif name == 'resnet50':
        net = ResNet50()
    elif name == 'resnet101':
        net = ResNet101()
    elif name == 'resnet152':
        net = ResNet152()
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()
    
    return net 

def make_folders(args):
    folders = [
        f"./results/{args.outdir}",
        f"./results/{args.outdir}/logdir/{args.mode}",
        f"./results/{args.outdir}/checkpoints",
        f"./results/{args.outdir}/logs"
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)


def get_model_parameters_number(model: nn.Module,
                                units: Optional[str] = None,
                                precision: int = 2):
    """Calculate parameter number of a model.

    Args:
        model (nn.module): The model for parameter number calculation.

    Returns:
        float: Parameter number of the model.
    """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params_string = params_to_string(num_params, units=units, precision=precision)
    print(f"model: {model.__class__.__name__}\t Params: {num_params_string}")
    return num_params


def params_to_string(num_params: float,
                     units: Optional[str] = None,
                     precision: int = 2) -> str:
    """Convert parameter number into a string.

    Args:
        num_params (float): Parameter number to be converted.
        units (str | None): Converted FLOPs units. Options are None, 'M',
            'K' and ''. If set to None, it will automatically choose the most
            suitable unit for Parameter number. Default: None.
        precision (int): Digit number after the decimal point. Default: 2.

    Returns:
        str: The converted parameter number with units.

    Examples:
        >>> params_to_string(1e9)
        '1000.0 M'
        >>> params_to_string(2e5)
        '200.0 k'
        >>> params_to_string(3e-9)
        '3e-09'
    """
    if units is None:
        if num_params // 10**6 > 0:
            return str(round(num_params / 10**6, precision)) + ' M'
        elif num_params // 10**3:
            return str(round(num_params / 10**3, precision)) + ' k'
        else:
            return str(num_params)
    else:
        if units == 'M':
            return str(round(num_params / 10.**6, precision)) + ' ' + units
        elif units == 'K':
            return str(round(num_params / 10.**3, precision)) + ' ' + units
        else:
            return str(num_params)



