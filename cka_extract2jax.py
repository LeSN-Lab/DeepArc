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
from flax import linen as nn
# parser
parser = argparse.ArgumentParser(description='Process modularity.')
parser.add_argument('--base_dir', type=str,
                    help= 'Where the trained model will be saved')
parser.add_argument('--gpu', type=int,
                    help= 'gpu limit')
args = parser.parse_args()
#(gpu connection - automatically connection)

# model_load, weights
model = MyModel()
with open('model_weights.pkl', 'rb') as f:
    loaded_params = pickle.load(f)

layers = []
cka_dir = os.path.join(args.base_dir, 'cka_within_model_256_normalize_activations.pkl')
with open(cka_dir, 'rb') as f:
    cka = pickle.load(f)
print(cka)

# layer processing
cka1 = cka[layers]
cka1 = cka1[:, layers]
print(cka.shape, cka1.shape)



out_dir = os.path.join(args.base_dir, 'cka_within_model_256_b.pkl')
with open(out_dir, 'wb') as f:
  pickle.dump(cka1, f)

# Plotting, plot_ckalist_resume
plot_dir = os.path.join(args.base_dir, 'layer')
plot_ckalist_resume([cka],plot_dir)

plot_dir = os.path.join(args.base_dir, 'layerEven')
plot_ckalist_resume([cka[::2,::2]],plot_dir)

plot_dir = os.path.join(args.base_dir, 'layerOdd')
plot_ckalist_resume([cka[1::2,1::2]],plot_dir)

plot_dir = os.path.join(args.base_dir, 'block')
plot_ckalist_resume([cka1],plot_dir)
