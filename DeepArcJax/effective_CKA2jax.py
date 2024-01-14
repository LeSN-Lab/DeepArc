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
def result(hsic_accumulator, across_models=False, hsic_accumulator_model1=None, hsic_accumulator_model2=None):
    mean_hsic = hsic_accumulator
    if across_models:
        normalization1 = jnp.sqrt(hsic_accumulator_model1)
        normalization2 = jnp.sqrt(hsic_accumulator_model2)
        mean_hsic /= normalization1[:, None]
        mean_hsic /= normalization2[None, :]
    else:
        normalization = jnp.sqrt(jnp.diag(mean_hsic))
        mean_hsic /= normalization[:, None]  # HISC(K,K)
        mean_hsic /= normalization[None, :]  # HISC(L,L)
    return mean_hsic



# Model load

# Model relation

# workflow example
# 1. num_layer =
# 2. hsic_accumulator =
# 3. hsic_accumulator = update_state(hsic_accumulator, activations)
# 4. cka_result = result(hsic_accumulator)
# 5. print(cka_result)