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
    gram -= means[1:0]
    gram -= means[:-1]
    return gram.reshape(-1)


# update_state in jax





# result in jax