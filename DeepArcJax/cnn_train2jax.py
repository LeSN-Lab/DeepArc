import os
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import jax
from jax import numpy as jnp
import optax
from tqdm.auto import tqdm
import flax
from flax import linen as nn
from flax.training import train_state
import dm_pix as pix # pip install dm-pix
import os
import pickle

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    print(e)

#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# JAX doen't ship with any data loading functionality.
base_dir = "train_set/dataset/training_set"
batch_size = 64


training_set = tf.keras.utils.image_dataset_from_directory(
    base_dir, validation_split=0.2, batch_size=batch_size, subset="training", seed=5603
)

print("이거 확인할거야 0아니면 좋겠따.",len(training_set))

validation_set = tf.keras.utils.image_dataset_from_directory(
    base_dir,validation_split=0.2,batch_size=batch_size,subset="validation",seed=5603,
)

eval_set = tf.keras.utils.image_dataset_from_directory(
    "train_set/dataset/test_set", batch_size=batch_size
)

IMG_SIZE = 128

resize_and_rescale = tf.keras.Sequential(
    [
        tf.keras.layers.Resizing(IMG_SIZE, IMG_SIZE),
        tf.keras.layers.Rescaling(1.0 / 255),
    ]
)

rng = jax.random.PRNGKey(0)
rng, inp_rng, init_rng = jax.random.split(rng, 3)

delta = 0.42
factor = 0.42

@jax.jit
def data_augmentation(image):
    new_image = pix.adjust_brightness(image=image, delta=delta)
    new_image = pix.random_brightness(image=new_image, max_delta=delta, key=inp_rng)
    new_image = pix.flip_up_down(image=image)
    new_image = pix.flip_left_right(image=new_image)
    new_image = pix.rot90(k=1, image=new_image) # k = number of times the rotation is applied

    return new_image

# plt.figure(figsize=(10, 10))
# augmented_images = []
# for images, _ in training_set.take(1):
#  for i in range(9):
#    augmented_image = data_augmentation(np.array(images[i], dtype=jnp.float32))
#    augmented_images.append(augmented_image)
#    ax = plt.subplot(3, 3, i + 1)
#    plt.imshow(augmented_images[i].astype("uint8"))
#    plt.axis("off")

jit_data_augmentation = jax.vmap(data_augmentation)

AUTOTUNE = tf.data.AUTOTUNE


def prepare(ds, shuffle=False):
    # Rescale and resize all datasets.
    ds = ds.map(lambda x, y: (resize_and_rescale(x), y), num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(1000)

    # Use buffered prefetching on all datasets.
    return ds.prefetch(buffer_size=AUTOTUNE)


train_ds = prepare(training_set, shuffle=True)
val_ds = prepare(validation_set)
evaluation_set = prepare(eval_set)

def get_batches(ds):
    data = ds.prefetch(1)
    # tfds.dataset_as_numpy converts the tf.data.Dataset into an iterable of NumPy arrays
    return tfds.as_numpy(data)

training_data = get_batches(train_ds)
validation_data = get_batches(val_ds)
evaluation_data = get_batches(evaluation_set)

class_names = training_set.class_names
num_classes = len(class_names)
print("num_classes:",num_classes)

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
        x = nn.Dense(features=num_classes)(x)
        return x

    def get_activations(self):
        return self.temp_list


model = CNN()
inp = jnp.ones([1, IMG_SIZE, IMG_SIZE, 3])
    # Initialize the model
params = model.init(init_rng, inp)
# print(params)


learning_rate = 1e-5
optimizer = optax.adam(
    learning_rate=learning_rate
)  # lr 1e-4. try 0.001 the default in tf.keras.optimizers.Adam
model_state = train_state.TrainState.create(
    apply_fn=model.apply, params=params, tx=optimizer
)
def calculate_loss_acc(state, params, batch):
    data_input, labels = batch
    data_input = jit_data_augmentation(data_input)
    # Obtain the logits and predictions of the model for the input data
    logits = state.apply_fn(params, data_input)
    # Calculate the loss and accuracy
    labels_onehot = jax.nn.one_hot(labels, num_classes=num_classes)
    #uncomment the line below for multiclass classification
    # loss = optax.softmax_cross_entropy(logits, labels_onehot).mean()
    loss = optax.sigmoid_binary_cross_entropy(logits, labels_onehot).mean()
    # comment the line above for multiclass problems
    acc = jnp.mean(jnp.argmax(logits, -1) == labels)
    return loss, acc

# print(training_data)
batch = next(iter(training_data))
calculate_loss_acc(model_state, model_state.params, batch)

with jax.disable_jit():
    @jax.jit  # Jit the function for efficiency
    def train_step(state, batch):
        # Gradient function
        grad_fn = jax.value_and_grad(
            calculate_loss_acc,  # Function to calculate the loss
            argnums=1,  # Parameters are second argument of the function
            has_aux=True,  # Function has additional outputs, here accuracy
        )
        # Determine gradients for current model, parameters and batch
        (loss, acc), grads = grad_fn(state, state.params, batch)
        # Perform parameter update with gradients and optimizer
        state = state.apply_gradients(grads=grads)
        # Return state and any other value we might want
        return state, loss, acc

    @jax.jit  # Jit the function for efficiency
    def eval_step(state, batch):
        # Determine the accuracy
        loss, acc = calculate_loss_acc(state, state.params, batch)
        return loss, acc
    training_accuracy = []
    training_loss = []

    testing_loss = []
    testing_accuracy = []


    def train_model(state, train_loader, test_loader, num_epochs=30):
        # Training loop
        for epoch in tqdm(range(num_epochs)):
            train_batch_loss, train_batch_accuracy = [], []
            val_batch_loss, val_batch_accuracy = [], []

            for train_batch in train_loader:
                state, loss, acc = train_step(state, train_batch)
                train_batch_loss.append(loss)
                train_batch_accuracy.append(acc)

            for val_batch in test_loader:
                val_loss, val_acc = eval_step(state, val_batch)

                val_batch_loss.append(val_loss)
                val_batch_accuracy.append(val_acc)

            # Loss for the current epoch
            epoch_train_loss = np.mean(train_batch_loss)
            epoch_val_loss = np.mean(val_batch_loss)

            # Accuracy for the current epoch
            epoch_train_acc = np.mean(train_batch_accuracy)
            epoch_val_acc = np.mean(val_batch_accuracy)

            testing_loss.append(epoch_val_loss)
            testing_accuracy.append(epoch_val_acc)

            training_loss.append(epoch_train_loss)
            training_accuracy.append(epoch_train_acc)

            print(
                f"Epoch: {epoch + 1}, loss: {epoch_train_loss:.2f}, acc: {epoch_train_acc:.2f} val loss: {epoch_val_loss:.2f} val acc {epoch_val_acc:.2f} "
            )

        return state

    trained_model_state = train_model(
            model_state, training_data, validation_data, num_epochs=2
        )

metrics_df = pd.DataFrame(np.array(training_accuracy), columns=["accuracy"])
metrics_df["val_accuracy"] = np.array(testing_accuracy)
metrics_df["loss"] = np.array(training_loss)
metrics_df["val_loss"] = np.array(testing_loss)
metrics_df[["loss", "val_loss"]].plot()
metrics_df[["accuracy", "val_accuracy"]].plot()

## 모델 저장

with open('model.pkl', 'wb') as f:
    pickle.dump(model,f)


from flax.training import checkpoints

checkpoints.save_checkpoint(
    ckpt_dir="/home/deeparc/DeepArc/DeepArcJax/content/my_checkpoints/",  # Folder to save checkpoint in
    target=trained_model_state,  # What to save. To only save parameters, use model_state.params
    step=100,  # Training step or other metric to save best model on
    prefix="my_model",  # Checkpoint file name prefix
    overwrite=True,  # Overwrite existing checkpoint files
)

loaded_model_state = checkpoints.restore_checkpoint(
    ckpt_dir="/home/deeparc/DeepArc/DeepArcJax/content/my_checkpoints/",  # Folder with the checkpoints
    target=model_state,  # (optional) matching object to rebuild state in
    prefix="my_model",  # Checkpoint file name prefix
)


