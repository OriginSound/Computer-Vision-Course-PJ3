import torch 
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

import numpy as np 
from aug import Cutmix, Cutout, Mixup  


class BaselineDataset(Dataset):
    def __init__(self, train=True):
        normalize = transforms.Normalize(
           mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
           std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        if train:
            transforms_ = transform_train
        else:
            transforms_ = transform_test

        self.dataset = torchvision.datasets.CIFAR100(
            "./cifar_data", 
            train=train, 
            transform=transforms_,
            download=True)
        self.length = len(self.dataset)
            
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        image, label = self.dataset[index]
        target = torch.zeros(100)
        target[label] = 1
        return image, target
    

class CutOutDataset(Dataset):
    def __init__(self, prob=0.8):
        self.cutout = Cutout()
        self.prob = prob

        self.dataset = BaselineDataset(train=True)
        self.length = len(self.dataset)
            
    def __len__(self):
        return self.length 
    
    def __getitem__(self, index):
        image, target = self.dataset[index]

        if np.random.rand() < self.prob:
            image, target = self.cutout((image, target))
        
        return image, target
    

class MixUpDataset(Dataset):
    def __init__(self, prob=0.8):
        self.mixup = Mixup()
        self.prob = prob

        self.dataset = BaselineDataset(train=True)
        self.length = len(self.dataset)
            
    def __len__(self):
        return self.length 
    
    def __getitem__(self, index):
        image, target = self.dataset[index]

        if np.random.rand() < self.prob:
            image_, target_ = self.dataset[np.random.randint(self.length)]
            image, target = self.mixup((image, target), (image_, target_))

        return image, target 


class CutMixDataset(Dataset):
    def __init__(self, prob=0.8):
        self.cutmix = Cutmix()
        self.prob = prob

        self.dataset = BaselineDataset(train=True)
        self.length = len(self.dataset)
            
    def __len__(self):
        return self.length 
    
    def __getitem__(self, index):
        image, target = self.dataset[index]

        if np.random.rand() < self.prob:
            image_, target_ = self.dataset[np.random.randint(self.length)]
            image, target = self.cutmix((image, target), (image_, target_))

        return image, target 


        