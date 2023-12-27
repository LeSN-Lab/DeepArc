'''
input: cka_dir 、threshold
output: module results divided by layer and block respectively
[Specific division] Number of modules and number of layers
'''
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import os
import argparse

# parser
parser = argparse.ArgumentParser(description='Process modularity.')
parser.add_argument('--base_dir', type=str,
                    help= 'Where the trained model will be saved')
parser.add_argument('--gpu', type=int,
                    help= 'gpu limit')
args = parser.parse_args()

# gpu connection

gpus =


# pickle_load



# cka_dir


# plot_dir

