# =========================================
# RGB Fashion-MNIST GAN
# =========================================

import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ------------------------------
# 1. Directories
# ------------------------------
OUTPUT_DIR = "fashion_rgb_gan"
os.makedirs(OUTPUT_DIR, exist_ok=True)
GENERATOR_PATH = os.path.join(OUTPUT_DIR, "generator_fashion_rgb.keras")
GENERATED_DIR = os.path.join(OUTPUT_DIR, "generated_patterns")
os.makedirs(GENERATED_DIR, exist_ok=True)

# ------------------------------
# 2. Load Fashion-MNIST and preprocess
# ------------------------------
(train_images, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
train_images = train_images.reshape(-1,28,28,1).astype("float32")
train_images = np.repeat(train_images, 3, axis=-1)  # convert to RGB
train_images = (train_images - 127.5) / 127.5       # [-1,1]

BUFFER_SIZE = train_images.shape[0]
BATCH_SIZE = 64
dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# ------------------------------
# 3. Generator
# ------------------------------
LATENT_DIM = 100
CHANNELS = 3

def build_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256,use_bias=False,input_shape=(LATENT_DIM,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((7,7,256)))

    model.add(layers.Conv2DTranspose(128,(5,5),strides=(1,1),padding="same",use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64,(5,5),strides=(2,2),padding="same",use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(CHANNELS,(5,5),strides=(2,2),padding="same",use_bias=False,activation="tanh"))
    return model

# ------------------------------
# 4. Discriminator
# ------------------------------
def build_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64,(5,5),strides=(2,2),padding="same",input_shape=[28,28,CHANNELS]))
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
def discriminator_loss(real,fake):
    return cross_entropy(tf.ones_like(real),real)+cross_entropy(tf.zeros_like(fake),fake)
def generator_loss(fake):
    return cross_entropy(tf.ones_like(fake),fake)

generator = build_generator()
discriminator = build_discriminator()

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# ------------------------------
# 6. Training
# ------------------------------
EPOCHS = 10
num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate,LATENT_DIM])

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE,LATENT_DIM])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise,training=True)
        real_output = discriminator(images,training=True)
        fake_output = discriminator(generated_images,training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output,fake_output)
    gradients_of_generator = gen_tape.gradient(gen_loss,generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss,discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator,generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,discriminator.trainable_variables))

def generate_and_save_images(model,epoch,test_input):
    predictions = model(test_input,training=False)
    fig, axes = plt.subplots(4,4,figsize=(8,8))
    axes = axes.flatten()
    for i,img in enumerate(predictions):
        img = (img+1)/2.0
        axes[i].imshow(img)
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,f"epoch_{epoch:03d}.png"))
    plt.close()

def train(dataset,epochs):
    for epoch in range(1,epochs+1):
        for batch in dataset:
            train_step(batch)
        generate_and_save_images(generator,epoch,seed)
        print(f"✅ Epoch {epoch}/{epochs} completed.")
    # Save generator
    generator.save(GENERATOR_PATH)
    print(f"💾 Generator saved at {GENERATOR_PATH}")

# ------------------------------
# 7. Inference
# ------------------------------
def generate_images(n=16):
    noise = tf.random.normal([n,LATENT_DIM])
    images = generator(noise,training=False)
    fig, axes = plt.subplots((n+3)//4,4,figsize=(12,12))
    axes = axes.flatten()
    for i,img in enumerate(images):
        axes[i].imshow((img+1)/2.0)
        axes[i].axis("off")
        # Save each image
        img_uint8 = (img.numpy()*255).astype(np.uint8)
        Image.fromarray(img_uint8).save(os.path.join(GENERATED_DIR,f"pattern_{i+1}.png"))
    plt.show()
    print(f"✅ {n} patterns saved in {GENERATED_DIR}/")

# ------------------------------
# 8. Run
# ------------------------------
if __name__=="__main__":
    train(dataset,EPOCHS)
    generate_images(16)
