from flax import linen as nn


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

import pickle

with open(file='model.pkl', mode='rb') as f:
    model=pickle.load(f)

print(model.get_activations())