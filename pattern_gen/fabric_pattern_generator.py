import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import zipfile
import cv2

# Optional import: try kagglehub but don't fail if unavailable
try:
    import kagglehub
except Exception:
    kagglehub = None

# Parameters
IMG_SIZE = 128
LATENT_DIM = 100
BATCH_SIZE = 32
EPOCHS = 2
CHANNELS = 3  # RGB images

# Directories
os.makedirs('fabric_patterns', exist_ok=True)
os.makedirs('generated_patterns', exist_ok=True)
os.makedirs('training_progress', exist_ok=True)

# ---------- Dataset download / fallback ----------
def download_fabric_dataset():
    """
    Try to download dataset using kagglehub (if available).
    If that fails, return None so that fallback sample patterns are created.
    If kagglehub returns a zip path, extract and return the folder path.
    """
    if kagglehub is None:
        print("kagglehub not available in environment. Using fallback patterns.")
        return None

    try:
        print("Attempting Kaggle dataset download via kagglehub...")
        path = kagglehub.dataset_download("alexanderliao/fabric-patterns")
        print("Download returned:", path)
        # if a zip file was returned, extract it
        if isinstance(path, str) and path.endswith('.zip') and os.path.exists(path):
            extract_dir = 'fabric_patterns_raw'
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(path, 'r') as z:
                z.extractall(extract_dir)
            print("Extracted to:", extract_dir)
            return extract_dir
        elif isinstance(path, str) and os.path.isdir(path):
            return path
        else:
            # Unexpected return type -> fallback
            print("Unexpected return from kagglehub. Using fallback patterns.")
            return None
    except Exception as e:
        print("Kaggle download failed:", e)
        return None

# Fallback method to create sample patterns if download fails
def create_sample_patterns():
    print("Creating sample patterns (fallback)...")
    patterns = []

    # Stripes
    for i in range(50):
        img = np.zeros((IMG_SIZE, IMG_SIZE, CHANNELS), dtype=np.uint8)
        color1 = [np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255)]
        color2 = [np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255)]
        for j in range(IMG_SIZE):
            if j % 16 < 8:
                img[j, :, :] = color1
            else:
                img[j, :, :] = color2
        patterns.append(img)
        Image.fromarray(img).save(f'fabric_patterns/stripe_{i}.png')

    # Checkerboard
    for i in range(50):
        img = np.zeros((IMG_SIZE, IMG_SIZE, CHANNELS), dtype=np.uint8)
        color1 = [np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255)]
        color2 = [np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255)]
        for x in range(IMG_SIZE):
            for y in range(IMG_SIZE):
                if (x // 16 + y // 16) % 2 == 0:
                    img[x, y, :] = color1
                else:
                    img[x, y, :] = color2
        patterns.append(img)
        Image.fromarray(img).save(f'fabric_patterns/checker_{i}.png')

    # Floral patterns (simplified)
    for i in range(50):
        img = np.zeros((IMG_SIZE, IMG_SIZE, CHANNELS), dtype=np.uint8)
        bg_color = [np.random.randint(150, 255), np.random.randint(150, 255), np.random.randint(150, 255)]
        flower_color = [np.random.randint(0, 150), np.random.randint(0, 150), np.random.randint(0, 150)]
        img[:, :, :] = bg_color
        for _ in range(30):
            center_x = np.random.randint(0, IMG_SIZE)
            center_y = np.random.randint(0, IMG_SIZE)
            radius = np.random.randint(5, 15)
            cv2.circle(img, (center_y, center_x), radius, flower_color, -1)  # note (col, row)
        patterns.append(img)
        Image.fromarray(img).save(f'fabric_patterns/floral_{i}.png')

    # Geometric patterns
    for i in range(50):
        img = np.zeros((IMG_SIZE, IMG_SIZE, CHANNELS), dtype=np.uint8)
        color1 = [np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255)]
        color2 = [np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255)]
        for x in range(IMG_SIZE):
            for y in range(IMG_SIZE):
                if (x * y) % 24 < 12:
                    img[x, y, :] = color1
                else:
                    img[x, y, :] = color2
        patterns.append(img)
        Image.fromarray(img).save(f'fabric_patterns/geometric_{i}.png')

    print("Sample patterns created:", len(patterns))
    return patterns

# Load and preprocess the dataset
def load_dataset():
    dataset_folder = download_fabric_dataset()
    images = []

    if dataset_folder is None:
        # fallback: programmatic patterns returned as list of numpy arrays
        images = create_sample_patterns()
    else:
        # Load images from the downloaded/extracted dataset folder
        for root, dirs, files in os.walk(dataset_folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        img_path = os.path.join(root, file)
                        img = Image.open(img_path).convert('RGB')
                        img = img.resize((IMG_SIZE, IMG_SIZE))
                        img = np.array(img)
                        images.append(img)
                    except Exception as e:
                        print(f"Error loading image {file}: {e}")

    if len(images) == 0:
        raise RuntimeError("No images found or created. Check dataset path or fallback creation.")

    # Convert to numpy array and normalize to [-1, 1]
    images = np.array(images, dtype=np.float32)
    images = (images - 127.5) / 127.5

    print(f"Loaded {len(images)} fabric patterns")
    return images

# ---------- Models ----------
def build_generator():
    model = tf.keras.Sequential(name="generator")
    model.add(layers.Input(shape=(LATENT_DIM,)))
    model.add(layers.Dense(8 * 8 * 512, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((8, 8, 512)))

    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(CHANNELS, (5, 5), strides=(1, 1), padding='same',
                                     use_bias=False, activation='tanh'))
    return model

def build_discriminator():
    model = tf.keras.Sequential(name="discriminator")
    model.add(layers.Input(shape=(IMG_SIZE, IMG_SIZE, CHANNELS)))
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))  # logits
    return model

# Losses & optimizers
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Prepare the dataset
def prepare_dataset():
    images = load_dataset()
    dataset = tf.data.Dataset.from_tensor_slices(images)
    dataset = dataset.shuffle(buffer_size=len(images)).batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return dataset

# Training step (works with variable batch sizes)
@tf.function
def train_step(images, generator, discriminator, gen_opt, disc_opt):
    batch_size = tf.shape(images)[0]
    noise = tf.random.normal([batch_size, LATENT_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_opt.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    disc_opt.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

# Training loop
def train(dataset, epochs, generator, discriminator, gen_opt, disc_opt):
    step = 0
    for epoch in range(epochs):
        print(f"Starting epoch {epoch+1}/{epochs}")
        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch, generator, discriminator, gen_opt, disc_opt)
            step += 1

            if step % 50 == 0:
                print(f"Step {step}: gen_loss={gen_loss:.4f}, disc_loss={disc_loss:.4f}")

        # After each epoch, generate a small fixed sample to monitor progress
        seed = tf.random.normal([16, LATENT_DIM])
        generate_and_save_images(generator, epoch + 1, seed)
        print(f"Epoch {epoch + 1} finished. gen_loss={gen_loss:.4f}, disc_loss={disc_loss:.4f}")

# Generate and save images for monitoring
def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)  # shape [N, H, W, C]
    # Convert from [-1,1] to [0,1]
    preds = (predictions + 1.0) / 2.0
    preds = tf.clip_by_value(preds, 0.0, 1.0).numpy()

    fig = plt.figure(figsize=(4, 4))
    for i in range(preds.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(preds[i])
        plt.axis('off')
    out_path = os.path.join('training_progress', f'image_at_epoch_{epoch:04d}.png')
    plt.savefig(out_path)
    plt.close()
    print("Saved sample grid to:", out_path)

# Generate patterns for user (save to disk)
def generate_patterns(model, n=16):
    n = int(n)
    noise = tf.random.normal([n, LATENT_DIM])
    generated_images = model(noise, training=False)
    generated_images = (generated_images + 1.0) * 127.5
    generated_images = tf.cast(generated_images, tf.uint8).numpy()

    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(-1)

    for i in range(len(axes)):
        ax = axes[i]
        if i < n:
            ax.imshow(generated_images[i])
            ax.set_title(f"Pattern {i+1}")
            ax.axis('off')
            Image.fromarray(generated_images[i]).save(f"generated_patterns/pattern_{i}.png")
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.show()
    print(f"{n} patterns saved to generated_patterns/ directory")

# ---------- Main ----------
def main():
    generator = build_generator()
    discriminator = build_discriminator()

    gen_optimizer = tf.keras.optimizers.Adam(1e-4)
    disc_optimizer = tf.keras.optimizers.Adam(1e-4)

    weights_path = 'fabric_pattern_generator.weights.h5'

    # Try to load saved weights (if they exist)
    if os.path.exists(weights_path):
        try:
            generator.load_weights(weights_path)
            print("Loaded generator weights from", weights_path)
        except Exception as e:
            print("Failed to load generator weights:", e)
            print("Starting training from scratch.")
            dataset = prepare_dataset()
            train(dataset, EPOCHS, generator, discriminator, gen_optimizer, disc_optimizer)
            generator.save_weights(weights_path)
            print("Saved generator weights to", weights_path)
    else:
        # Train from scratch
        print("No existing weights found. Training new model.")
        dataset = prepare_dataset()
        train(dataset, EPOCHS, generator, discriminator, gen_optimizer, disc_optimizer)
        generator.save_weights(weights_path)
        print("Saved generator weights to", weights_path)

    # Interactively ask user for number to generate
    while True:
        try:
            n = int(input("How many patterns to generate? (1-16): "))
            if 1 <= n <= 16:
                generate_patterns(generator, n)
                break
            else:
                print("Please enter a number between 1 and 16")
        except ValueError:
            print("Please enter a valid number")

    # Optionally generate more
    while True:
        more = input("Generate more patterns? (y/n): ").strip().lower()
        if more == 'y':
            try:
                n = int(input("How many patterns to generate? (1-16): "))
                if 1 <= n <= 16:
                    generate_patterns(generator, n)
                else:
                    print("Please enter a number between 1 and 16")
            except ValueError:
                print("Please enter a valid number")
        elif more == 'n':
            break
        else:
            print("Please enter 'y' or 'n'")

if __name__ == "__main__":
    main()
