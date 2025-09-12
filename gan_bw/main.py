# ==============================
# Fashion-MNIST GAN Training Script (with Model Save/Load)
# ==============================

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # suppress remapper warnings

import sys
import subprocess
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import imageio
from glob import glob

# Suppress TensorFlow optimizer logs
tf.get_logger().setLevel("ERROR")

# ------------------------------
# 1. Setup output directory
# ------------------------------
OUTPUT_DIR = "gan_outputs"
MODEL_PATH = os.path.join(OUTPUT_DIR, "generator_fashion_mnist.h5")
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

    model.add(layers.Reshape((7, 7, 256)))
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding="same", use_bias=False, activation="tanh"))

    return model

# ------------------------------
# 4. Discriminator
# ------------------------------
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same",
                            input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
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

generator = make_generator_model()
discriminator = make_discriminator_model()

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# ------------------------------
# 6. Training loop
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

    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow((predictions[i, :, :, 0] + 1) / 2.0, cmap="gray")
        plt.axis("off")

    filename = os.path.join(OUTPUT_DIR, f"epoch_{epoch:03d}.png")
    plt.savefig(filename)
    plt.close()

def train(dataset, epochs):
    for epoch in range(1, epochs+1):
        for image_batch in dataset:
            train_step(image_batch)

        # Save images every 5 epochs, plus first and last
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            generate_and_save_images(generator, epoch, seed)

        print(f"✅ Epoch {epoch}/{epochs} completed.")

    # Save trained generator at the end
    generator.save(MODEL_PATH)
    print(f"💾 Trained generator saved at {MODEL_PATH}")

# ------------------------------
# 7. GIF Maker (auto-open)
# ------------------------------
def create_gif(output_dir="gan_outputs", gif_name="training_progress.gif"):
    frames = []
    files = sorted(glob(os.path.join(output_dir, "epoch_*.png")))
    for file in files:
        frames.append(imageio.imread(file))

    gif_path = os.path.join(output_dir, gif_name)

    if frames:
        imageio.mimsave(gif_path, frames, fps=5)
        print(f"🎞️ GIF saved at {gif_path}")

        # Auto-open depending on OS
        try:
            if sys.platform.startswith("darwin"):   # macOS
                subprocess.call(["open", gif_path])
            elif os.name == "nt":                   # Windows
                os.startfile(gif_path)
            elif os.name == "posix":                # Linux
                subprocess.call(["xdg-open", gif_path])
        except Exception as e:
            print(f"⚠️ Could not auto-open GIF: {e}")

    else:
        print("⚠️ No images found to make GIF.")

# ------------------------------
# 8. Run Training
# ------------------------------
if __name__ == "__main__":
    train(dataset, EPOCHS)
    create_gif(OUTPUT_DIR)
