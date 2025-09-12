# ==============================
# Fashion-MNIST GAN Inference Script (Interactive + Random Colormap Mode)
# ==============================

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tensorflow.keras.models import load_model
from PIL import Image
import time
import random

# ------------------------------
# Config
# ------------------------------
OUTPUT_DIR = "gan_inference"
MODEL_PATH = "gan_outputs/generator_fashion_mnist.h5"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LATENT_DIM = 100
IMAGES_PER_BATCH = 16  # grid size

# ------------------------------
# Load Trained Generator
# ------------------------------
print("🔄 Loading trained generator...")
generator = load_model(MODEL_PATH)
print("✅ Generator loaded successfully.")

# ------------------------------
# Generate One Batch of Images (Colormap)
# ------------------------------
def generate_images(batch_idx, cmap_name, n=IMAGES_PER_BATCH, save=True):
    noise = tf.random.normal([n, LATENT_DIM])
    generated_images = generator(noise, training=False)

    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    axes = np.array(axes).reshape(-1)

    cmap = cm.get_cmap(cmap_name)

    for i in range(n):
        # Convert [-1,1] → [0,1]
        img = (generated_images[i, :, :, 0] + 1) / 2.0  

        # Apply colormap → RGBA
        rgba_img = cmap(img)

        # Drop alpha channel → RGB
        rgb_img = (rgba_img[:, :, :3] * 255).astype(np.uint8)

        # Show image
        axes[i].imshow(rgb_img)
        axes[i].axis("off")

        if save:
            filename = os.path.join(OUTPUT_DIR, f"{cmap_name}_batch{batch_idx}_img{i+1}.png")
            Image.fromarray(rgb_img).save(filename)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

    if save:
        print(f"💾 Batch {batch_idx}: {n} {cmap_name} RGB images saved in {OUTPUT_DIR}/")

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    try:
        # Ask for colormap
        cmap_name = input("Enter colormap (default=plasma, or type 'random'): ").strip() or "plasma"
        
        # Get all available colormaps
        all_cmaps = plt.colormaps()

        # Ask for number of batches
        num_batches = int(input("How many batches do you want to generate? (each = 16 images): "))
        if num_batches > 0:
            for b in range(1, num_batches+1):
                if cmap_name == "random":
                    chosen_cmap = random.choice(all_cmaps)
                else:
                    chosen_cmap = cmap_name if cmap_name in all_cmaps else "plasma"

                print(f"\n🚀 Generating batch {b}/{num_batches} with colormap '{chosen_cmap}'...")
                generate_images(b, chosen_cmap)
                time.sleep(1)
        else:
            print("⚠️ Please enter a positive number.")
    except ValueError:
        print("⚠️ Invalid input. Please enter a number.")
