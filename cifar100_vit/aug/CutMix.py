import torch 
import numpy as np 

from torch import Tensor
from typing import Tuple


class Cutmix(object):
    """Randomly cut out one patche from one image 
       and mix up with another image.
    Args:
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, length=20):
        self.length = length

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

        _, h, w = img_1.shape 

        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask = torch.ones_like(img_1)
        mask[:, y1:y2, x1:x2] = 0 
        
        img = mask * img_1 + (1-mask) * img_2 
        prob = 1 - (x2-x1)*(y2-y1)/ (h*w) 
        label = prob * label_1 + (1-prob) * label_2
        return img, label 
    


