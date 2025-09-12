# =========================================
# RGB Fabric GAN - Training on Synthetic Data
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
import cv2
from glob import glob

# Suppress TF warnings
tf.get_logger().setLevel("ERROR")
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"

# ------------------------------
# 1. Directories & Paths
# ------------------------------
OUTPUT_DIR = "rgb_gan_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
GENERATOR_PATH = os.path.join(OUTPUT_DIR, "generator_rgb.h5")
DISCRIMINATOR_PATH = os.path.join(OUTPUT_DIR, "discriminator_rgb.h5")

# ------------------------------
# 2. Generate synthetic RGB dataset
# ------------------------------
IMG_SIZE = 32
CHANNELS = 3
NUM_PER_PATTERN = 50

def create_synthetic_dataset():
    images = []

    # Stripes
    for _ in range(NUM_PER_PATTERN):
        img = np.zeros((IMG_SIZE,IMG_SIZE,CHANNELS),dtype=np.uint8)
        color1 = np.random.randint(50,255,3)
        color2 = np.random.randint(50,255,3)
        for j in range(IMG_SIZE):
            img[j,:,:] = color1 if j%8<4 else color2
        images.append(img)

    # Checkerboard
    for _ in range(NUM_PER_PATTERN):
        img = np.zeros((IMG_SIZE,IMG_SIZE,CHANNELS),dtype=np.uint8)
        color1 = np.random.randint(50,255,3)
        color2 = np.random.randint(50,255,3)
        for x in range(IMG_SIZE):
            for y in range(IMG_SIZE):
                img[x,y,:] = color1 if (x//4 + y//4)%2==0 else color2
        images.append(img)

    # Simple floral dots
    for _ in range(NUM_PER_PATTERN):
        img = np.zeros((IMG_SIZE,IMG_SIZE,CHANNELS),dtype=np.uint8)
        bg = np.random.randint(150,255,3)
        dot = np.random.randint(0,150,3)
        img[:,:,:] = bg
        for _ in range(10):
            cx,cy = np.random.randint(0,IMG_SIZE,2)
            r = np.random.randint(2,5)
            cv2.circle(img,(cx,cy),r,dot,-1)
        images.append(img)

    # Geometric patterns
    for _ in range(NUM_PER_PATTERN):
        img = np.zeros((IMG_SIZE,IMG_SIZE,CHANNELS),dtype=np.uint8)
        color1 = np.random.randint(50,255,3)
        color2 = np.random.randint(50,255,3)
        for x in range(IMG_SIZE):
            for y in range(IMG_SIZE):
                img[x,y,:] = color1 if (x*y)%8<4 else color2
        images.append(img)

    images = np.array(images,dtype=np.float32)
    images = (images - 127.5)/127.5   # normalize [-1,1]
    print(f"✅ Synthetic dataset created: {images.shape}")
    return images

dataset_images = create_synthetic_dataset()
BUFFER_SIZE = dataset_images.shape[0]
BATCH_SIZE = 32
dataset = tf.data.Dataset.from_tensor_slices(dataset_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# ------------------------------
# 3. Generator
# ------------------------------
LATENT_DIM = 100
def build_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(4*4*256,use_bias=False,input_shape=(LATENT_DIM,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((4,4,256)))

    model.add(layers.Conv2DTranspose(128,(5,5),strides=(2,2),padding="same",use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64,(5,5),strides=(2,2),padding="same",use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(32,(5,5),strides=(2,2),padding="same",use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(CHANNELS,(5,5),strides=(1,1),padding="same",use_bias=False,activation="tanh"))
    return model

# ------------------------------
# 4. Discriminator
# ------------------------------
def build_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64,(5,5),strides=(2,2),padding="same",input_shape=[IMG_SIZE,IMG_SIZE,CHANNELS]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128,(5,5),strides=(2,2),padding="same"))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256,(5,5),strides=(2,2),padding="same"))
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
    return cross_entropy(tf.ones_like(real),real) + cross_entropy(tf.zeros_like(fake),fake)
def generator_loss(fake):
    return cross_entropy(tf.ones_like(fake),fake)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# ------------------------------
# 6. Load or create models
# ------------------------------
if os.path.exists(GENERATOR_PATH) and os.path.exists(DISCRIMINATOR_PATH):
    generator = load_model(GENERATOR_PATH)
    discriminator = load_model(DISCRIMINATOR_PATH)
    print("✅ Loaded existing models.")
else:
    generator = build_generator()
    discriminator = build_discriminator()
    print("⚠️ New models created.")

# ------------------------------
# 7. Training functions
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
    fig = plt.figure(figsize=(4,4))
    for i in range(predictions.shape[0]):
        plt.subplot(4,4,i+1)
        img = (predictions[i]+1)/2.0
        plt.imshow(img)
        plt.axis("off")
    plt.savefig(os.path.join(OUTPUT_DIR,f"epoch_{epoch:03d}.png"))
    plt.close()

def train(dataset,epochs):
    for epoch in range(1,epochs+1):
        for batch in dataset:
            train_step(batch)
        generate_and_save_images(generator,epoch,seed)
        print(f"✅ Epoch {epoch}/{epochs} completed.")

    # Save models
    generator.save(GENERATOR_PATH)
    discriminator.save(DISCRIMINATOR_PATH)
    print("💾 Generator and Discriminator saved.")

# ------------------------------
# 8. Create GIF
# ------------------------------
def create_gif(output_dir=OUTPUT_DIR,gif_name="training.gif"):
    frames=[]
    files=sorted(glob(os.path.join(output_dir,"epoch_*.png")))
    for f in files:
        frames.append(imageio.imread(f))
    if frames:
        gif_path=os.path.join(output_dir,gif_name)
        imageio.mimsave(gif_path,frames,fps=5)
        print(f"🎞️ GIF saved at {gif_path}")
        try:
            if sys.platform.startswith("darwin"):
                subprocess.call(["open",gif_path])
            elif os.name=="nt":
                os.startfile(gif_path)
            elif os.name=="posix":
                subprocess.call(["xdg-open",gif_path])
        except: pass

# ------------------------------
# 9. Optional inference
# ------------------------------
def generate_images(n=16):
    noise = tf.random.normal([n,LATENT_DIM])
    images = generator(noise,training=False)
    for i in range(n):
        img = (images[i]+1)/2.0
        plt.imshow(img)
        plt.axis("off")
        plt.show()

# ------------------------------
# 10. Run
# ------------------------------
if __name__=="__main__":
    if not (os.path.exists(GENERATOR_PATH) and os.path.exists(DISCRIMINATOR_PATH)):
        train(dataset,EPOCHS)
    else:
        print("✅ Models already exist. Skipping training.")
    create_gif()
    # Optional: generate new patterns
    # generate_images(5)
