import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random
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
epochs_gan = 5
epochs_classifier = 5
noise_dim = 100
num_gen_images_per_class = 100
num_classes = 10
gan_output_dir = "gan_images_pipeline"
os.makedirs(gan_output_dir, exist_ok=True)

# -----------------------
# Dataset
# -----------------------
transform = transforms.ToTensor()

class TensorFashionMNIST(datasets.FashionMNIST):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, torch.tensor(target, dtype=torch.long)

train_dataset = TensorFashionMNIST(root="data", train=True, download=True, transform=transform)
test_dataset = TensorFashionMNIST(root="data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

classes = ["T-shirt/top","Trouser","Pullover","Dress","Coat",
           "Sandal","Shirt","Sneaker","Bag","Ankle boot"]

# -----------------------
# Conditional GAN
# -----------------------
class Generator(nn.Module):
    def __init__(self, noise_dim=100, num_classes=10):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(noise_dim+num_classes,256*7*7),
            nn.BatchNorm1d(256*7*7),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1,(256,7,7)),
            nn.ConvTranspose2d(256,128,5,1,2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128,64,5,2,2,output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64,1,5,2,2,output_padding=1),
            nn.Tanh()
        )
    def forward(self, noise, labels):
        c = self.label_emb(labels)
        x = torch.cat([noise,c],dim=1)
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self,num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.model = nn.Sequential(
            nn.Conv2d(2, 64, 5, 2, 2),  # 1 image channel + 1 label channel
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Conv2d(64, 128, 5, 2, 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(128*7*7, 1)
        )

    def forward(self,img,labels):
        label_map = labels.float().unsqueeze(1).unsqueeze(2).unsqueeze(3)
        label_map = label_map.expand(-1,1,img.size(2), img.size(3)) / (self.num_classes-1)
        x = torch.cat([img, label_map], dim=1)
        return self.model(x)

# -----------------------
# Initialize GAN
# -----------------------
generator = Generator(noise_dim=noise_dim,num_classes=num_classes).to(device)
discriminator = Discriminator(num_classes=num_classes).to(device)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=1e-4)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
adversarial_loss = nn.BCEWithLogitsLoss()

# -----------------------
# Train Conditional GAN
# -----------------------
print("=== Training Conditional GAN ===")
for epoch in range(epochs_gan):
    for real_imgs, labels in train_loader:
        real_imgs, labels = real_imgs.to(device), labels.to(device)
        batch_size_curr = real_imgs.size(0)
        valid = torch.ones(batch_size_curr,1,device=device)
        fake = torch.zeros(batch_size_curr,1,device=device)

        # Generator
        optimizer_G.zero_grad()
        z = torch.randn(batch_size_curr,noise_dim,device=device)
        random_labels = torch.randint(0,num_classes,(batch_size_curr,),device=device)
        gen_imgs = generator(z, random_labels)
        g_loss = adversarial_loss(discriminator(gen_imgs, random_labels), valid)
        g_loss.backward()
        optimizer_G.step()

        # Discriminator
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(real_imgs, labels), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), random_labels), fake)
        d_loss = (real_loss + fake_loss)/2
        d_loss.backward()
        optimizer_D.step()
    print(f"[cGAN] Epoch {epoch+1}/{epochs_gan} | D loss: {d_loss.item():.4f} | G loss: {g_loss.item():.4f}")

# -----------------------
# Generate labeled GAN images for augmentation
# -----------------------
generator.eval()
all_images, all_labels = [], []
with torch.no_grad():
    for label in range(num_classes):
        z = torch.randn(num_gen_images_per_class, noise_dim, device=device)
        labels_tensor = torch.full((num_gen_images_per_class,), label, dtype=torch.long, device=device)
        gen_imgs = generator(z, labels_tensor)
        gen_imgs = (gen_imgs + 1)/2  # scale to [0,1]
        all_images.append(gen_imgs.cpu())
        all_labels.append(labels_tensor.cpu())

all_images = torch.cat(all_images, dim=0)
all_labels = torch.cat(all_labels, dim=0)
torch.save((all_images, all_labels), os.path.join(gan_output_dir,"gan_images_labeled.pt"))
print(f"✅ Generated conditional GAN images saved at {gan_output_dir}/gan_images_labeled.pt")

# -----------------------
# Train Classifier on real + GAN images
# -----------------------
print("=== Training Classifier ===")
classifier = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28,512),
    nn.ReLU(),
    nn.Linear(512,512),
    nn.ReLU(),
    nn.Linear(512,num_classes)
).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)

gan_images, gan_labels = torch.load(os.path.join(gan_output_dir,"gan_images_labeled.pt"))
gan_dataset = TensorDataset(gan_images, gan_labels)
augmented_dataset = ConcatDataset([train_dataset, gan_dataset])
augmented_loader = DataLoader(augmented_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(epochs_classifier):
    classifier.train()
    for X,y in augmented_loader:
        X,y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = classifier(X)
        loss = loss_fn(logits,y)
        loss.backward()
        optimizer.step()
    
    # Evaluate on real test images
    classifier.eval()
    correct,total = 0,0
    with torch.no_grad():
        for X_test, y_test in DataLoader(test_dataset, batch_size=batch_size):
            X_test, y_test = X_test.to(device), y_test.to(device)
            preds = classifier(X_test).argmax(dim=1)
            correct += (preds==y_test).sum().item()
            total += y_test.size(0)
    print(f"[Classifier] Epoch {epoch+1}/{epochs_classifier} | Test Accuracy: {100*correct/total:.2f}%")

# -----------------------
# Visualize predictions on real Fashion-MNIST test images
# -----------------------
classifier.eval()
plt.figure(figsize=(8,8))
num_samples = 16
indices = random.sample(range(len(test_dataset)), num_samples)

for i, idx in enumerate(indices):
    x, y_actual = test_dataset[idx]
    x_device = x.unsqueeze(0).to(device)
    with torch.no_grad():
        y_pred_idx = classifier(x_device).argmax(dim=1).item()
        y_pred = classes[y_pred_idx]
        y_actual_label = classes[y_actual]

    plt.subplot(4, 4, i+1)
    plt.imshow(x.squeeze(), cmap="gray")
    color = "red" if y_pred != y_actual_label else "black"
    plt.title(f"P: {y_pred}\nA: {y_actual_label}", color=color, fontsize=9)
    plt.axis("off")

plt.tight_layout()
plt.show()
