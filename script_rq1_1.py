
import os
dir = 'train/resnet152/'
commend = 'CUDA_VISIBLE_DEVICES=0 python cifar_train.py --base_dir {}'.format(dir)
os.system(dir)
