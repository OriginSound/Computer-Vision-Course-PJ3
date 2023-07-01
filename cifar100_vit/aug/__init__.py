from .CutOut import Cutout
from .CutMix import Cutmix
from .MixUp import Mixup


def show():
    # 1, three kinds of augmentations
    cutout = Cutout()
    cutmix = Cutmix()
    mixup = Mixup()

    # 2, datasets
    import torchvision
    from torchvision import transforms
    train_data = torchvision.datasets.CIFAR100(
        "./cifar_data", 
        train=True, 
        transform=transforms.ToTensor(),
        download=True
    )
    raw = []

    cutmix_list = []
    cutout_list = []
    mixup_list = []

    # 3, record
    data0 = train_data[0] 
    for i in range(1, 3+1):
        data = train_data[i]
        raw.append(data[0])
        # Cutout
        res = cutout(data)
        cutout_list.append(res[0])
        # Mixup
        res = mixup(data, data0)
        mixup_list.append(res[0])
        # Cutmix 
        res = cutmix(data, data0)
        cutmix_list.append(res[0])
        import pdb; pdb.set_trace()

    import torch 
    import torchvision    
    raw = torch.stack(raw)
    cutout_list = torch.stack(cutout_list)
    mixup_list = torch.stack(mixup_list)
    cutmix_list = torch.stack(cutmix_list)
    images = torch.cat([raw, cutout_list, mixup_list, cutmix_list])
    images = torchvision.utils.make_grid(images, nrow=3)
    torchvision.utils.save_image(images, "./results/imgs/0.png")



    
    
