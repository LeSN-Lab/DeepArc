import jax
import jax.numpy as jnp
from flax import linen as nn
from PIL import Image
import numpy as np
import optax

from flax.training import checkpoints
from flax.training import train_state


# Define the CNN model
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

IMG_SIZE = 128
# Function to preprocess the image
def preprocess_image(image_path, target_size=(IMG_SIZE, IMG_SIZE)):
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size)
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def load_model(ckpt_path):
    try:
        # Load the checkpoint file
        ckpt_data = np.load(ckpt_path, allow_pickle=True)
        if isinstance(ckpt_data, np.ndarray) and ckpt_data.dtype == 'object':
            ckpt_dict = ckpt_data.item()
            params = jax.tree_map(lambda x: jax.numpy.array(x), ckpt_dict)
        else:
            params = jax.numpy.array(ckpt_data)
        return params
    except Exception as e:
        print("Error loading the checkpoint:", e)
        return None

ckpt_path = '/home/deeparc/DeepArc/DeepArcJax/content/my_checkpoints/my_model100/checkpoint'
# Path to your checkpoint and image
# ckpt_path = '/Users/songhaein/PycharmProjects/DeepArc/my_checkpoints/my_model100/checkpoint'  # Change this to your checkpoint path
image_path = '/home/deeparc/DeepArc/DeepArcJax/test1.jpg'
image_path2 = '/home/deeparc/DeepArc/DeepArcJax/test2.jpg'
# Upload and set path to your image

# Initialize the model

# Load the trained model parameters
#params = load_model(ckpt_path)

# Preprocess the image
preprocessed_image1 = preprocess_image(image_path)
preprocessed_image2 = preprocess_image(image_path2)

# Predict function
@jax.jit
def predict(params, image):
    return model.apply({'params': params}, image)



model = CNN()



inp = jnp.ones([1, IMG_SIZE, IMG_SIZE, 3])
rng = jax.random.PRNGKey(0)
rng, inp_rng, init_rng = jax.random.split(rng, 3)
params = model.init(init_rng, inp)

learning_rate = 1e-5
optimizer = optax.adam(
    learning_rate=learning_rate
)  # lr 1e-4. try 0.001 the default in tf.keras.optimizers.Adam
model_state = train_state.TrainState.create(
    apply_fn=model.apply, params=params, tx=optimizer
)

loaded_model_state = checkpoints.restore_checkpoint(
    ckpt_dir="/home/deeparc/DeepArc/DeepArcJax/content/my_checkpoints/",  # Folder with the checkpoints
    target=model_state,  # (optional) matching object to rebuild state in
    prefix="my_model",  # Checkpoint file name prefix
)


print(model.get_activations())
print("========================================")
# Make a prediction
predictions = model.apply(loaded_model_state.params,preprocessed_image1)
print(model.get_activations())
print("========================================")
predictions = model.apply(loaded_model_state.params,preprocessed_image2)
print(model.get_activations())
print("========================================")



# Process the predictions as needed
# For example, if it's a classification model, you might want to find the class with the highest probability
predicted_class = np.argmax(predictions, axis=1)

print("Predicted Class:", predicted_class)
