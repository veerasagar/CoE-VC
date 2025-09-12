# =========================================
# Fashion-MNIST GAN with Model Save/Load
# =========================================

import os
import sys
import subprocess
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import imageio
from glob import glob

# Suppress TensorFlow optimizer logs
tf.get_logger().setLevel("ERROR")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # suppress remapper warnings

# ------------------------------
# 1. Setup directories
# ------------------------------
OUTPUT_DIR = "gan_outputs"
GENERATOR_PATH = os.path.join(OUTPUT_DIR, "generator_fashion_mnist.h5")
DISCRIMINATOR_PATH = os.path.join(OUTPUT_DIR, "discriminator_fashion_mnist.h5")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------
# 2. Load & preprocess Fashion-MNIST
# ------------------------------
(train_images, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype("float32")
train_images = (train_images - 127.5) / 127.5  # normalize to [-1, 1]

BUFFER_SIZE = 60000
BATCH_SIZE = 256
dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# ------------------------------
# 3. Generator
# ------------------------------
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((7,7,256)))
    model.add(layers.Conv2DTranspose(128,(5,5),strides=(1,1),padding="same",use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(64,(5,5),strides=(2,2),padding="same",use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(1,(5,5),strides=(2,2),padding="same",use_bias=False,activation="tanh"))
    return model

# ------------------------------
# 4. Discriminator
# ------------------------------
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64,(5,5),strides=(2,2),padding="same",input_shape=[28,28,1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128,(5,5),strides=(2,2),padding="same"))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model

# ------------------------------
# 5. Loss & Optimizers
# ------------------------------
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# ------------------------------
# 6. Load models if exist
# ------------------------------
if os.path.exists(GENERATOR_PATH) and os.path.exists(DISCRIMINATOR_PATH):
    generator = load_model(GENERATOR_PATH)
    discriminator = load_model(DISCRIMINATOR_PATH)
    print("✅ Loaded existing Generator and Discriminator models.")
else:
    generator = make_generator_model()
    discriminator = make_discriminator_model()
    print("⚠️ No saved models found. New models created.")

# ------------------------------
# 7. Training
# ------------------------------
EPOCHS = 5
noise_dim = 100
num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, noise_dim])

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4,4))
    for i in range(predictions.shape[0]):
        plt.subplot(4,4,i+1)
        plt.imshow((predictions[i,:,:,0]+1)/2.0, cmap="gray")
        plt.axis("off")
    filename = os.path.join(OUTPUT_DIR,f"epoch_{epoch:03d}.png")
    plt.savefig(filename)
    plt.close()

def train(dataset, epochs):
    for epoch in range(1, epochs+1):
        for image_batch in dataset:
            train_step(image_batch)

        if epoch % 1 == 0:
            generate_and_save_images(generator, epoch, seed)

        print(f"✅ Epoch {epoch}/{epochs} completed.")

    # Save models
    generator.save(GENERATOR_PATH)
    discriminator.save(DISCRIMINATOR_PATH)
    print(f"💾 Generator and Discriminator saved at {OUTPUT_DIR}.")

# ------------------------------
# 8. Create GIF
# ------------------------------
def create_gif(output_dir="gan_outputs", gif_name="training_progress.gif"):
    frames = []
    files = sorted(glob(os.path.join(output_dir,"epoch_*.png")))
    for file in files:
        frames.append(imageio.imread(file))

    gif_path = os.path.join(output_dir, gif_name)

    if frames:
        imageio.mimsave(gif_path, frames, fps=5)
        print(f"🎞️ GIF saved at {gif_path}")
        try:
            if sys.platform.startswith("darwin"):
                subprocess.call(["open", gif_path])
            elif os.name=="nt":
                os.startfile(gif_path)
            elif os.name=="posix":
                subprocess.call(["xdg-open", gif_path])
        except:
            pass
    else:
        print("⚠️ No images found to make GIF.")

# ------------------------------
# 9. Optional inference
# ------------------------------
def generate_images(n=16):
    noise = tf.random.normal([n, noise_dim])
    images = generator(noise, training=False)
    for i in range(n):
        img = (images[i]+1)/2.0
        plt.imshow(img[:,:,0], cmap="gray")
        plt.axis("off")
        plt.show()

# ------------------------------
# 10. Run script
# ------------------------------
if __name__=="__main__":
    # Train only if models do not exist
    if not (os.path.exists(GENERATOR_PATH) and os.path.exists(DISCRIMINATOR_PATH)):
        train(dataset, EPOCHS)
    else:
        print("✅ Models already exist. Skipping training.")

    create_gif(OUTPUT_DIR)
    # Optional: generate a few images
    # generate_images(5)
