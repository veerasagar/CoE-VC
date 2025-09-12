# RVCE-VC-CoE

## FashionGAN-Classifier

It is an end-to-end deep learning pipeline that combines Conditional GANs and a Neural Network classifier to enhance Fashion-MNIST image classification. The cGAN generates labeled synthetic fashion images that augment the real dataset, allowing the classifier to learn from both real and generated images. The system evaluates on real test images, displaying predicted versus actual labels, and highlights misclassifications, demonstrating how synthetic data can improve model performance in image recognition tasks.

## Project Overview

1. **Conditional GAN (cGAN)**  
   - Generates fashion images conditioned on specific labels (T-shirt, Trouser, etc.).  
   - Produces synthetic labeled images to augment the real dataset.  

2. **Classifier**  
   - A simple fully connected neural network trained on **real + GAN-generated images**.  
   - Evaluates on real Fashion-MNIST test images.  

3. **Visualization**  
   - Displays predictions on real test images.  
   - Shows **Predicted (P)** vs **Actual (A)** labels, highlighting incorrect predictions in red.  

## Dataset

- **Fashion-MNIST**: 28x28 grayscale images of clothing items.
- Classes:  
  `T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot`

## Installation

```bash
pip install -r requirements.txt
python main.py
```
