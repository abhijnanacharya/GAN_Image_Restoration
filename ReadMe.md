# Image Restoration Using GANs

This project implements a Generative Adversarial Network (GAN) for image restoration tasks, specifically designed to work with the CelebA dataset. The GAN aims to reconstruct high-quality images from degraded inputs caused by blur, noise, or both.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Project Structure](#project-structure)
5. [Usage](#usage)
6. [Results](#results)
7. [References](#references)

---

## Overview

The project includes:

- A **Generator** network for reconstructing degraded images.
- A **Discriminator** network for distinguishing real images from restored ones.
- A custom dataset class for handling the CelebA dataset and applying degradations such as blur and noise.

The training process uses a combination of adversarial loss and pixel-wise L1 loss to improve the generator's output quality.

---

## Features

- **Customizable degradations**:
  - Blur
  - Noise
  - Combined (blur + noise)
- Configurable training settings in `training_config.py`.
- Save generated samples and model checkpoints at defined intervals.
- Modular and reusable code structure for GAN training.

---

## Requirements

### Prerequisites

- Python 3.8+
- PyTorch 1.9.0+
- torchvision
- Pillow
- NumPy

Install dependencies using:

`pip install -r requirements.txt`

## Project Structure

Here is the structure of the project:

```
IMAGE_RESTORATION_GAN/
├── config/
│   └── training_config.py       # Training hyperparameters
├── data/
│   └── celeba/
│       ├── img_align_celeba/    # CelebA images (downloaded separately)
│       ├── list_eval_partition.txt  # Train/val/test split information
│       ├── ...                  # Other CelebA metadata files
├── dataset/
│   └── celeba_dataset.py        # Custom CelebA dataset class
├── models/
│   ├── generator.py             # Generator network
│   ├── discriminator.py         # Discriminator network
│   └── residual_block.py        # Residual block for the generator
├── training/
│   ├── train.py                 # Main training loop
│   └── stats.py                 # Helper class to track training statistics
├── utils/
│   ├── checkpoint_utils.py      # Functions for saving/loading checkpoints
│   ├── image_utils.py           # Utilities for saving sample images
│   ├── plot_utils.py            # Optional: Visualization utilities
├── output/                      # Directory for outputs (samples and checkpoints)
├── .gitignore                   # Git ignore file
├── main.py                      # Main entry point (optional, calls train.py)
├── requirements.txt             # Required Python packages
└── README.md                    # Project documentation
```

---

---

## Usage

### 1. Prepare the Dataset

Download the CelebA dataset and place the files under the `data/celeba/` directory as shown in the structure above.

### 2. Configure Training

Modify the training parameters in `config/training_config.py` as needed. Key parameters include:

- Number of epochs
- Batch size
- Learning rate
- Weight for pixel loss (`lambda_pixel`)

### 3. Train the Model

Run the training script:
`python main.py`

### 4. View Results

- Generated sample images will be saved under `output/<run_timestamp>/samples/`.
- Checkpoints will be saved under `output/<run_timestamp>/checkpoints/`.

---

## Results

The GAN reconstructs high-quality images from degraded inputs. Example results:

| Low Resolution (Degraded) | Generated (Restored) | Ground Truth |
| ------------------------- | -------------------- | ------------ |
|                           |                      |              |
