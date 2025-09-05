"""
data_loader.py - Comprehensive data loading utilities for fashion design AI

This module provides data loaders for various fashion datasets including:
- Fashion-MNIST
- DeepFashion
- Custom garment pattern datasets
- 3D garment datasets
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
from PIL import Image
from typing import Optional, Tuple, Dict

class FashionMNISTLoader:
    """Data loader for Fashion-MNIST dataset"""

    def __init__(
        self,
        data_dir: str = './data',
        batch_size: int = 128,
        image_size: int = 28,
        augment: bool = True
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size

        # Define transformations
        if augment:
            self.transform_train = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        else:
            self.transform_train = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

        self.transform_test = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.class_names = [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]

    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Get train and test data loaders"""
        train_dataset = FashionMNIST(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.transform_train
        )
        test_dataset = FashionMNIST(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.transform_test
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=4, pin_memory=True
        )
        return train_loader, test_loader


class CustomFashionDataset(Dataset):
    """Custom dataset for fashion images and patterns"""

    def __init__(
        self,
        data_dir: str,
        metadata_file: str,
        transform: Optional[transforms.Compose] = None,
        include_patterns: bool = False
    ):
        self.data_dir = data_dir
        self.transform = transform
        self.include_patterns = include_patterns

        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        self.image_paths = list(self.metadata.keys())

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        full_path = os.path.join(self.data_dir, image_path)
        image = Image.open(full_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        meta = self.metadata[image_path]
        sample = {
            'image': image,
            'category': meta.get('category', 0),
            'color': meta.get('color', ''),
            'style': meta.get('style', ''),
            'season': meta.get('season', ''),
        }
        if self.include_patterns and 'pattern' in meta:
            sample['pattern'] = torch.tensor(meta['pattern'], dtype=torch.float32)
        return sample


class PatternDataset(Dataset):
    """Dataset for garment sewing patterns"""

    def __init__(self, pattern_dir: str, pattern_format: str = 'dxf'):
        self.pattern_dir = pattern_dir
        self.format = pattern_format
        self.pattern_files = [
            f for f in os.listdir(pattern_dir)
            if f.endswith(f'.{pattern_format}')
        ]

    def __len__(self):
        return len(self.pattern_files)

    def __getitem__(self, idx):
        file = self.pattern_files[idx]
        path = os.path.join(self.pattern_dir, file)
        if self.format == 'json':
            with open(path, 'r') as f:
                data = json.load(f)
        else:
            import ezdxf
            doc = ezdxf.readfile(path)
            msp = doc.modelspace()
            data = []
            for e in msp:
                if e.dxftype() == 'LINE':
                    data.append({
                        'type': 'line',
                        'start': e.dxf.start,
                        'end': e.dxf.end
                    })
                # Add other entity types as needed
        return {'pattern_data': data, 'filename': file}


def create_fashion_loaders(config: Dict) -> Dict[str, DataLoader]:
    """
    Factory to create data loaders:
    config = {
      'dataset': 'fashion_mnist'|'deepfashion'|'custom',
      'data_dir': './data',
      'batch_size': 128,
      ...
    }
    """
    loaders = {}
    ds = config['dataset']
    if ds == 'fashion_mnist':
        loader = FashionMNISTLoader(
            data_dir=config['data_dir'],
            batch_size=config['batch_size'],
            image_size=config.get('image_size', 28),
            augment=config.get('augment', True)
        )
        loaders['train'], loaders['test'] = loader.get_loaders()
    elif ds == 'custom':
        transform = transforms.Compose([
            transforms.Resize((config.get('image_size', 256),) * 2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        ds = CustomFashionDataset(
            data_dir=config['data_dir'],
            metadata_file=config['custom_metadata'],
            transform=transform,
            include_patterns=config.get('include_patterns', False)
        )
        loaders['custom'] = DataLoader(
            ds, batch_size=config['batch_size'], shuffle=True,
            num_workers=config.get('num_workers', 4)
        )
    # Add DeepFashion loader as needed
    return loaders
