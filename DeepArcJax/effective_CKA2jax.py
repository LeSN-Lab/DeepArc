import jax
import jax.numpy as jnp
from jax import jit

# jax: 상태 저장 부분을 정의하기 위해 Flax의 nn.Module 사용
class MinibatchCKA(nn.Module):
    num_layers: int
    num_layers2: int
    across_models: bool = False

    def setup(self):
        self.hsic_accumulator = self.param('hsic_accumulator',
                                           nn.initializers.zeros,
                                           (self.num_layers, self.num_layers2))

        if self.across_models:
            self.hsic_accumulator_model1 = self.param('hsic_accumulator_model1',
                                                      nn.initializers.zeros,
                                                      (self.num_layers,))
            self.hsic_accumulator_model2 = self.param('hsic_accumulator_model2',
                                                      nn.initializers.zeros,
                                                      (self.num_layers2,))

    def _generate_gram_matrix(self, x):
        x = x.reshape(x.shape[0], -1)
        gram = jnp.matmul(x, x.T)
        n = gram.shape[0]
        gram = jax.ops.index_update(gram, jax.ops.index.diag_indices(n), jnp.zeros(n))
        means = jnp.sum(gram, axis=0) / (n - 2)
        means -= jnp.sum(means) / (2 * (n - 1))
        gram -= means[:, None]
        gram -= means[None, :]
        gram = jax.ops.index_update(gram, jax.ops.index.diag_indices(n), jnp.zeros(n))
        return gram.ravel()

    def update_state(self, activations):
        layer_grams = jnp.stack([self._generate_gram_matrix(x) for x in activations])
        self.hsic_accumulator += jnp.matmul(layer_grams, layer_grams.T)

    def update_state_across_models(self, activations1, activations2):
        layer_grams1 = jnp.stack([self._generate_gram_matrix(x) for x in activations1])
        layer_grams2 = jnp.stack([self._generate_gram_matrix(x) for x in activations2])
        self.hsic_accumulator += jnp.matmul(layer_grams1, layer_grams2.T)
        self.hsic_accumulator_model1 += jnp.einsum('ij,ij->i', layer_grams1, layer_grams1)
        self.hsic_accumulator_model2 += jnp.einsum('ij,ij->i', layer_grams2, layer_grams2)

    def result(self):
        mean_hsic = self.hsic_accumulator  #(num_layers, num_layers2)
        if self.across_models:
            normalization1 = jnp.sqrt(self.hsic_accumulator_model1)
            normalization2 = jnp.sqrt(self.hsic_accumulator_model2)
            mean_hsic /= normalization1[:, None]
            mean_hsic /= normalization2[None, :]
        else:
            normalization = jnp.sqrt(jnp.diag(mean_hsic))
            mean_hsic /= normalization[:, None]  # HISC(K,K)
            mean_hsic /= normalization[None, :]  # HISC(L,L)
        return mean_hsic


# workflow example
# 1. num_layer =
# 2. hsic_accumulator =
# 3. hsic_accumulator = update_state(hsic_accumulator, activations)
# 4. cka_result = result(hsic_accumulator)
# 5. print(cka_result)