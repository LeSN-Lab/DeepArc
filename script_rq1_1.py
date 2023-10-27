import os

dir = 'train/resnet56'
if os.path.exists('./checkpoint/' + dir):
    pass
else:
    commend = 'CUDA_VISIBLE_DEVICES={} python cifar_train.py --base_dir {}'.format(
        0,dir)

