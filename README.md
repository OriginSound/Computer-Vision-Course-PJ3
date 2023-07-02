# Computer-Vision-Course-PJ3

## MoCo

## Vit
|  Model | Accuracy | checkpoint |
|:------:|:--------:|:----------:|
|resnet18| 79.9%    | <a href="https://pan.baidu.com/s/1pl4pcCXGqQ0u0oGs362mpQ">resnet18</a></td>|
|  vit   | 55.8%    | <a href="https://pan.baidu.com/s/1blPF_WNF1OwNDA80fG1J2w">vit</a></td> |

The passwords are **1111**.

### Training
```
python train.py --model ${MODEL} --epoch ${EPOCH} --batchsize ${BATCHSIZE} --gpu ${GPU_ID} --mode 3
```

### Test
```
python test.py --checkpoint ${CHECKPOINT_FILE} --batchsize ${BATCHSIZE} --gpu ${GPU_ID}
```

## Nerf
Please see <a href="https://github.com/ashawkey/torch-ngp">this repo</a> and follow the instructions in it.

<<<<<<< HEAD
The configuration of the environment of **colmap** is referred to <a href="https://zhuanlan.zhihu.com/p/397053413">this blog</a>.
=======
The configuration of the environment of **colmap** is referred to <a href="https://zhuanlan.zhihu.com/p/397053413">this blog</a>.
>>>>>>> 8569a29100ed4d4130b12b20e35e6eba20ae67e9
