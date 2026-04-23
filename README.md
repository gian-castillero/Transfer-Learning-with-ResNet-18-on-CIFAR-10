# Transfer Learning with ResNet-18 on CIFAR-10

Applying and fine-tuning a pretrained ResNet-18 model for image classification on CIFAR-10, with Grad-CAM saliency visualizations and a critical discussion of model interpretability.

## Overview

This project demonstrates transfer learning in two stages: first by replacing only the final classification head of a pretrained ResNet-18 (feature extraction), then by fine-tuning all parameters end-to-end on the new task. The pretrained model was originally trained on ImageNet (1,000 classes); the target task is CIFAR-10 (10 classes: animals, vehicles). Grad-CAM visualizations are used to inspect what regions of an image drive the model's predictions.

## Dataset

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html): 60,000 color 32×32 images across 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). Split: 80% train / 20% validation from the official training set; 10,000-image test set held out for final evaluation.

## Methods

### Stage 1 — Feature Extraction (Output Layer Only)

All ResNet-18 parameters are frozen (`requires_grad=False`). Only the final fully connected layer is replaced with `Linear(512, 10)` and trained. This treats the pretrained backbone as a fixed feature extractor.

**Training:** SGD, `lr=0.01`, `batch_size=64`, `epochs=5`

**Result: >40% validation accuracy**

The pretrained ImageNet representations, though not trained on CIFAR-10, already encode generalizable visual features (edges, textures, shapes) that support meaningful classification on a new dataset with only 5,120 additional trainable parameters.

### Stage 2 — Full Fine-Tuning

The pretrained ResNet-18 backbone is unfrozen and all parameters are updated jointly. A larger learning rate is used since the network starts from a good initialization and needs to adapt more broadly.

**Training:** SGD, `lr=0.1`, `batch_size=64`, `epochs=10`

**Result: >75% validation accuracy**

Full fine-tuning significantly outperforms feature extraction, as all layers adapt to the lower-resolution, domain-shifted CIFAR-10 images.

### Grad-CAM Saliency Visualization

Gradient Class Activation Maps (Grad-CAM) are computed for the original pretrained ResNet-18 on three cat images (`tuxedo_cat.jpg`, `kittens.jpg`, `dog_cat.jpg`), each with respect to the ImageNet `cat` class (index 281). Grad-CAM computes a weighted average of the final convolutional layer's feature maps — weighted by the gradient of the target class score — and projects this as a heatmap onto the original image.

## Tech Stack

- Python 3
- PyTorch (`torch.nn`, `torch.optim`, `torchvision.models`)
- `pytorch-grad-cam`
- NumPy, Matplotlib, PIL

## How to Run

```bash
pip install torch torchvision grad-cam matplotlib pillow numpy jupyter
jupyter notebook transfer.ipynb
```

CIFAR-10 downloads automatically via `torchvision.datasets`. GPU acceleration (CUDA or MPS) is strongly recommended for fine-tuning. Cat images (`cat_images/`) must be present in the working directory.
