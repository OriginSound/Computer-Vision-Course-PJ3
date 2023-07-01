import torch
import numpy as np

from torch import Tensor 
from typing import Tuple 


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes=1, length=20):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, data: Tuple[Tensor, Tensor]):
        """
        Args:
            data: (image, label)
                image (Tensor): Tensor image of size (C, H, W)
                label (Tensor): (100, )
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        img, label = data 

        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float16)

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img, label