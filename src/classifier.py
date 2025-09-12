# classifier_fashion_mnist_conditional.py
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

batch_size = 128
epochs = 5
gan_images_path = "gan_images_conditional/gan_images_labeled.pt"

transform = transforms.ToTensor()

# Dataset class to ensure tensor labels
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

# Load conditional GAN images
if os.path.exists(gan_images_path):
    gan_images, gan_labels = torch.load(gan_images_path)
    gan_dataset = TensorDataset(gan_images, gan_labels)
    augmented_dataset = ConcatDataset([train_dataset, gan_dataset])
    print(f"Loaded {len(gan_dataset)} GAN images for augmentation")
else:
    augmented_dataset = train_dataset
    print("GAN images not found, training on real data only")

augmented_loader = DataLoader(augmented_dataset, batch_size=batch_size, shuffle=True)

# Classifier
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10)
        )
    def forward(self,x):
        x = self.flatten(x)
        return self.fc(x)

classifier = Classifier().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)

# Train Classifier
for epoch in range(epochs):
    classifier.train()
    for X,y in augmented_loader:
        X,y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = classifier(X)
        loss = loss_fn(logits,y)
        loss.backward()
        optimizer.step()
    
    # Test on real test set
    classifier.eval()
    correct,total = 0,0
    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test,y_test = X_test.to(device), y_test.to(device)
            preds = classifier(X_test).argmax(dim=1)
            correct += (preds==y_test).sum().item()
            total += y_test.size(0)
    print(f"[Classifier] Epoch {epoch+1}/{epochs} | Test Accuracy: {100*correct/total:.2f}%")

# Visualize some GAN images
classifier.eval()
plt.figure(figsize=(8,8))
if os.path.exists(gan_images_path):
    num_samples = 16
    indices = random.sample(range(len(gan_dataset)), num_samples)
    for i, idx in enumerate(indices):
        x = gan_dataset[idx][0].unsqueeze(0).to(device)
        with torch.no_grad():
            pred = classifier(x).argmax(dim=1).item()
        plt.subplot(4,4,i+1)
        plt.imshow(gan_dataset[idx][0][0], cmap="gray")
        plt.title(f"P: {classes[pred]}", fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    plt.show()
