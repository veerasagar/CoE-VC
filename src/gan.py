# gan_fashion_mnist.py
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# -----------------------
# Device
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# -----------------------
# Hyperparameters
# -----------------------
batch_size = 128
epochs = 5
noise_dim = 100
num_gen_images = 1000  # Number of images to generate
output_dir = "gan_images"
os.makedirs(output_dir, exist_ok=True)

# -----------------------
# Dataset
# -----------------------
transform = transforms.ToTensor()
train_dataset = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# -----------------------
# GAN Models
# -----------------------
class Generator(nn.Module):
    def __init__(self, noise_dim=100):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 256*7*7),
            nn.BatchNorm1d(256*7*7),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1,(256,7,7)),
            nn.ConvTranspose2d(256,128,5,stride=1,padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128,64,5,stride=2,padding=2,output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64,1,5,stride=2,padding=2,output_padding=1),
            nn.Tanh()
        )
    def forward(self,x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1,64,5,2,2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Conv2d(64,128,5,2,2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(128*7*7,1)
        )
    def forward(self,x):
        return self.model(x)

# -----------------------
# Loss & Optimizers
# -----------------------
adversarial_loss = nn.BCEWithLogitsLoss()
generator = Generator(noise_dim=noise_dim).to(device)
discriminator = Discriminator().to(device)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=1e-4)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

# -----------------------
# Train GAN
# -----------------------
for epoch in range(epochs):
    for real_imgs,_ in train_loader:
        real_imgs = real_imgs.to(device)
        batch_size_curr = real_imgs.size(0)
        valid = torch.ones(batch_size_curr,1,device=device)
        fake = torch.zeros(batch_size_curr,1,device=device)

        # Train generator
        optimizer_G.zero_grad()
        z = torch.randn(batch_size_curr, noise_dim, device=device)
        gen_imgs = generator(z)
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

        # Train discriminator
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss)/2
        d_loss.backward()
        optimizer_D.step()
    
    print(f"[GAN] Epoch {epoch+1}/{epochs} | D loss: {d_loss.item():.4f} | G loss: {g_loss.item():.4f}")

# -----------------------
# Generate Synthetic Images
# -----------------------
generator.eval()
with torch.no_grad():
    z = torch.randn(num_gen_images, noise_dim, device=device)
    gen_images = generator(z)
    gen_images = (gen_images + 1)/2  # rescale to [0,1]
    torch.save(gen_images.cpu(), os.path.join(output_dir,"gan_images.pt"))

print(f"✅ Generated {num_gen_images} images and saved to {output_dir}/gan_images.pt")
