import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from PIL import Image
import time

# ------------------------------
# Parameters
# ------------------------------
IMG_SIZE = 128
CHANNELS = 3  # RGB
LATENT_DIM = 100
BATCH_SIZE = 32
EPOCHS = 50
DATASET_DIR = "dataset"           # Put your colored images here
GENERATED_DIR = "generated_images"
os.makedirs(GENERATED_DIR, exist_ok=True)

# ------------------------------
# Load and preprocess dataset
# ------------------------------
def load_dataset():
    images = []
    for root, _, files in os.walk(DATASET_DIR):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                path = os.path.join(root, file)
                try:
                    img = Image.open(path).convert("RGB")
                    img = img.resize((IMG_SIZE, IMG_SIZE))
                    img = np.array(img, dtype=np.float32)
                    img = (img / 127.5) - 1  # normalize to [-1,1]
                    images.append(img)
                except:
                    pass
    images = np.array(images)
    print(f"Loaded {len(images)} images from {DATASET_DIR}")
    return images

# ------------------------------
# Build generator
# ------------------------------
def build_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(8*8*512, use_bias=False, input_shape=(LATENT_DIM,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((8,8,512)))

    # Upsample to 16x16
    model.add(layers.Conv2DTranspose(256, (5,5), strides=(2,2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Upsample to 32x32
    model.add(layers.Conv2DTranspose(128, (5,5), strides=(2,2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Upsample to 64x64
    model.add(layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Upsample to 128x128
    model.add(layers.Conv2DTranspose(32, (5,5), strides=(2,2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Output layer
    model.add(layers.Conv2DTranspose(CHANNELS, (5,5), strides=(1,1), padding='same', activation='tanh'))

    return model

# ------------------------------
# Build discriminator
# ------------------------------
def build_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5,5), strides=(2,2), padding='same', input_shape=(IMG_SIZE,IMG_SIZE,CHANNELS)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5,5), strides=(2,2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, (5,5), strides=(2,2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(512, (5,5), strides=(2,2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

# ------------------------------
# Loss functions
# ------------------------------
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# ------------------------------
# Prepare dataset
# ------------------------------
def prepare_dataset(images):
    dataset = tf.data.Dataset.from_tensor_slices(images)
    dataset = dataset.shuffle(len(images)).batch(BATCH_SIZE)
    return dataset

# ------------------------------
# Optimizers
# ------------------------------
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# ------------------------------
# Training step
# ------------------------------
@tf.function
def train_step(images, generator, discriminator):
    noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])

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

    return gen_loss, disc_loss

# ------------------------------
# Training loop
# ------------------------------
def train(dataset, generator, discriminator, epochs):
    for epoch in range(1, epochs+1):
        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch, generator, discriminator)

        print(f"Epoch {epoch}/{epochs}  Generator Loss: {gen_loss:.4f}  Discriminator Loss: {disc_loss:.4f}")

        # Save sample images every 5 epochs
        if epoch % 5 == 0:
            generate_images(generator, epoch, 4, save=True)

# ------------------------------
# Generate and save images
# ------------------------------
def generate_images(generator, epoch, n=16, save=True):
    noise = tf.random.normal([n, LATENT_DIM])
    generated_images = generator(noise, training=False)

    for i in range(n):
        img = (generated_images[i] + 1) * 127.5
        img = tf.cast(img, tf.uint8).numpy()
        filename = os.path.join(GENERATED_DIR, f"epoch{epoch}_img{i+1}.png")
        Image.fromarray(img).save(filename)

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    images = load_dataset()
    dataset = prepare_dataset(images)

    generator = build_generator()
    discriminator = build_discriminator()

    train(dataset, generator, discriminator, EPOCHS)

    # Final batch generation
    print("Generating final images...")
    generate_images(generator, epoch="final", n=16, save=True)
    print(f"All images saved in {GENERATED_DIR}/")
