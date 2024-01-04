# JAX/FLAX 모델 파라미터, 상태 관리가 다르게 작동. 모델의 파라미터는 명시적으로 관리, 상태 변경은 함수의 출력을 통해 처리됨. (tf.keras.Model)
# CKA 계산을 위한 로직또한 JAX의 함수형 프로그래밍 스타일로 재작성 되어야 함.

import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np
import os
import pickle
import argparse



parser = argparse.ArgumentParser(description='CKA calculation settings.')
parser.add_argument('--cka_batch', type=int, default=256, help='Batch size for CKA approximation')
parser.add_argument('--cka_iter', type=int, default=5, help='number of iterations for CKA approximation')
parser.add_argument('--experiment_name', type=str, default='None', help='Path to saved model')
args = parser.parse_args()

def normalize_activations(act):
    act = act.reshape(act.shape[0], -1)
    act_norm = jnp.linalg.norm(act, axis=1)
    return act / act_norm[:, None]

