import jax
import jax.numpy as jnp
from jax import jit


# MinibatchCKA in jax
def generate_gram_matrix(x):
    x = x.reshape((x.shape[0], -1))
    gram = jnp.dot(x, x.T)
    n = gram.shape[0]
    gram = gram.at[jnp.diag_indices(n)].set(0)
    means = jnp.sum(gram, axis=0) / (n - 2)
    means -= jnp.sum(means) / (2 * (n - 1))
    gram -= means[:, None]
    gram -= means[None, :]
    gram = gram.at[jnp.diag_indices(n)].set(0)
    return gram.reshape(-1)


# update_state in jax
def update_state(hsic_accumulator, activations):
    layer_grams = jnp.stack([generate_gram_matrix(x) for x in activations])
    hsic_accumulator += jnp.dot(layer_grams, layer_grams.T)
    return hsic_accumulator

# result in jax