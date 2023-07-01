import torch 
import numpy as np

from typing import Tuple
from torch import Tensor


class Mixup(object):
    """Mix up two images.
    Args:
        prob (float): the probability of the first image
    """
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, data1: Tuple[Tensor, Tensor], data2: Tuple[Tensor, Tensor]):
        """
        Args:
            data1 (Image: Tensor, Label: Tensor): (C, H, W), (100, ).
            data2 (Image: Tensor, Label: Tensor): (C, H, W), (100, ).
        Returns:
            data: 
        """
        img_1, label_1 = data1
        img_2, label_2 = data2 
        img = self.prob * img_1 + (1 - self.prob) * img_2 
        label = self.prob * label_1 + (1 - self.prob) * label_2

        return (img, label)