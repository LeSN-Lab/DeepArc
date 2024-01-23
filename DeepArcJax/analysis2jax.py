# JAX/FLAX 모델 파라미터, 상태 관리가 다르게 작동. 모델의 파라미터는 명시적으로 관리, 상태 변경은 함수의 출력을 통해 처리됨. (tf.keras.Model)
# CKA 계산을 위한 로직또한 JAX의 함수형 프로그래밍 스타일로 재작성 되어야 함.

import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np
import os
import pickle
import argparse
import effective_CKA2jax



parser = argparse.ArgumentParser(description='CKA calculation settings.')
parser.add_argument('--cka_batch', type=int, default=256, help='Batch size for CKA approximation')
parser.add_argument('--cka_iter', type=int, default=5, help='number of iterations for CKA approximation')
parser.add_argument('--experiment_name', type=str, default='None', help='Path to saved model')
args = parser.parse_args()

def normalize_activations(act):
    act = act.reshape(act.shape[0], -1)
    act_norm = jnp.linalg.norm(act, axis=1)
    return act / act_norm[:, None]

def get_activations(model):
    activations = []
    x = inputs
    for layer in model.layers:
        x = layer.apply({'params': params[layer.name]}, x)
        activations.append()
    return activations

# convert_bn_to_train_mode -> is_training flag 모델 정의 시)

##
class CNN(nn.Module):
    temp_list = []

    @nn.compact
    def __call__(self, x):
        self.temp_list.clear()
        x = nn.Conv(features=128, kernel_size=(3, 3))(x)
        self.temp_list.append(x)
        x = nn.relu(x)
        self.temp_list.append(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        self.temp_list.append(x)
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        self.temp_list.append(x)
        x = nn.relu(x)
        self.temp_list.append(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        self.temp_list.append(x)
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        self.temp_list.append(x)
        x = nn.relu(x)
        self.temp_list.append(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        self.temp_list.append(x)
        x = x.reshape((x.shape[0], -1))  # flatten
        self.temp_list.append(x)
        x = nn.Dense(features=256)(x)
        self.temp_list.append(x)
        x = nn.Dense(features=128)(x)
        self.temp_list.append(x)
        x = nn.relu(x)
        self.temp_list.append(x)
        x = nn.Dense(features=2)(x)
        return x

    def get_activations(self):
        return self.temp_list



model = CNN()






# compute CKA
# def compute_cka_internal(model_dir,
#                          data_path=None,
#                          dataset_name='cifar10',
#                          use_batch=True,
#                          use_train_mode=False,
#                          normalize_act=True):
#   if dataset_name == 'cifar10':
#     if use_train_mode:
#       filename = 'cka_within_model_%d_bn_train_mode.pkl' % FLAGS.cka_batch
#     else:
#       filename = 'cka_within_model_%d.pkl' % FLAGS.cka_batch
#   else:
#     suffix = dataset_name.split('_')[-1]
#     if use_train_mode:
#       filename = 'cka_within_model_%d_%s_bn_train_mode.pkl' % (FLAGS.cka_batch,
#                                                                suffix)
#     else:
#       filename = 'cka_within_model_%d_%s.pkl' % (FLAGS.cka_batch, suffix)
#   if normalize_act:
#     filename = filename.replace('.pkl', '_normalize_activations.pkl')
#   print('------------',model_dir)
#   out_dir = os.path.join(model_dir, filename)