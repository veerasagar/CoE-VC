import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
import zipfile

# Parameters
IMG_SIZE = 64
LATENT_DIM = 100
BATCH_SIZE = 32
EPOCHS = 200
CHANNELS = 3  # RGB images

# Create a directory for our dataset
os.makedirs('fashion_patterns', exist_ok=True)

# Download and prepare a sample dataset (using a small pattern dataset)
def download_sample_dataset():
    # This is a placeholder for your actual dataset
    # For a real project, you would use your own pattern images
    print("Preparing sample dataset...")
    
    # Create some simple pattern images programmatically
    for i in range(100):
        # Create a simple pattern
        img = np.zeros((IMG_SIZE, IMG_SIZE, CHANNELS), dtype=np.uint8)
        
        # Add different patterns based on index
        if i % 5 == 0:
            # Stripes
            for j in range(IMG_SIZE):
                if j % 8 < 4:
                    img[j, :, :] = [255, 100, 100]  # Red stripes
        elif i % 5 == 1:
            # Checkerboard
            for x in range(IMG_SIZE):
                for y in range(IMG_SIZE):
                    if (x // 8 + y // 8) % 2 == 0:
                        img[x, y, :] = [100, 255, 100]  # Green squares
        elif i % 5 == 2:
            # Dots
            for x in range(4, IMG_SIZE, 8):
                for y in range(4, IMG_SIZE, 8):
                    img[x-2:x+2, y-2:y+2, :] = [100, 100, 255]  # Blue dots
        elif i % 5 == 3:
            # Gradient
            for x in range(IMG_SIZE):
                for y in range(IMG_SIZE):
                    img[x, y, :] = [x*4, y*4, (x+y)*2]  # Color gradient
        else:
            # Mixed pattern
            for x in range(IMG_SIZE):
                for y in range(IMG_SIZE):
                    if (x + y) % 10 < 5:
                        img[x, y, :] = [255, 255, 100]  # Yellow pattern
                    else:
                        img[x, y, :] = [100, 255, 255]  # Cyan pattern
        
        # Save the image
        Image.fromarray(img).save(f'fashion_patterns/pattern_{i}.png')
    
    print("Sample dataset created!")

# Build the generator
def build_generator():
    model = tf.keras.Sequential()
    
    # Foundation for 8x8 image
    model.add(layers.Dense(8*8*256, use_bias=False, input_shape=(LATENT_DIM,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((8, 8, 256)))
    
    # Upsample to 16x16
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    # Upsample to 32x32
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    # Upsample to 64x64
    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    # Output layer
    model.add(layers.Conv2DTranspose(CHANNELS, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
    
    return model

# Build the discriminator
def build_discriminator():
    model = tf.keras.Sequential()
    
    # Input layer
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', 
                            input_shape=[IMG_SIZE, IMG_SIZE, CHANNELS]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    # Downsample to 32x32
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    # Downsample to 16x16
    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    # Classifier
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    
    return model

# Define the loss functions
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Create the models
generator = build_generator()
discriminator = build_discriminator()

# Define the optimizers
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Prepare the dataset
def prepare_dataset():
    download_sample_dataset()
    
    # Load the images
    image_paths = [f'fashion_patterns/pattern_{i}.png' for i in range(100)]
    images = []
    
    for path in image_paths:
        img = Image.open(path)
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img = np.array(img) / 127.5 - 1  # Normalize to [-1, 1]
        images.append(img)
    
    # Convert to numpy array
    images = np.array(images)
    
    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices(images)
    dataset = dataset.shuffle(100).batch(BATCH_SIZE)
    
    return dataset

# Training step
@tf.function
def train_step(images):
    # Generate noise
    noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generate images
        generated_images = generator(noise, training=True)
        
        # Discriminate real and fake images
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        # Calculate losses
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    
    # Calculate gradients
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    # Apply gradients
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return gen_loss, disc_loss

# Training function
def train(dataset, epochs):
    for epoch in range(epochs):
        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)
        
        # Generate images after each epoch
        if epoch % 10 == 0:
            generate_and_save_images(generator, epoch + 1, tf.random.normal([16, LATENT_DIM]))
            print(f'Epoch {epoch + 1}, Generator Loss: {gen_loss}, Discriminator Loss: {disc_loss}')

# Generate and save images
def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    
    fig = plt.figure(figsize=(4, 4))
    
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        # Convert from [-1, 1] to [0, 1]
        img = (predictions[i] + 1) / 2.0
        plt.imshow(img)
        plt.axis('off')
    
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()

# Generate new patterns
def generate_patterns(model, n=16):
    # Generate random noise
    noise = tf.random.normal([n, LATENT_DIM])
    
    # Generate images
    generated_images = model(noise, training=False)
    
    # Display the images
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        # Convert from [-1, 1] to [0, 1]
        img = (generated_images[i] + 1) / 2.0
        ax.imshow(img)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# Main function
def main():
    # Prepare dataset
    dataset = prepare_dataset()
    
    # Train the model
    print("Training the model...")
    train(dataset, EPOCHS)
    
    # Generate and display new patterns
    print("Generating new patterns...")
    generate_patterns(generator)
    
    # Save the generator model
    generator.save('fashion_pattern_generator.h5')
    print("Model saved as 'fashion_pattern_generator.h5'")

if __name__ == "__main__":
    main()