import os
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from utils.image_utils import save_generated_images
from utils.checkpoint_utils import save_checkpoint

def train_gan(config, generator, discriminator, train_loader, device):
    """
    Train the GAN model.
    Args:
        config: Training configuration object.
        generator: Generator model.
        discriminator: Discriminator model.
        train_loader: DataLoader for training data.
        device: Device to run training on (CPU/GPU).
    """
    criterion_gan = nn.BCELoss()
    criterion_pixel = nn.L1Loss()

    optimizer_G = optim.Adam(generator.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))

    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("output", f"run_{timestamp}")
    sample_dir = os.path.join(output_dir, "samples")
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(config.num_epochs):
        generator.train()
        discriminator.train()

        for i, (lr_images, hr_images) in enumerate(train_loader):
            batch_size = lr_images.size(0)

            # Prepare real and fake labels
            real_label = torch.ones(batch_size, 1).to(device) * 0.9
            fake_label = torch.zeros(batch_size, 1).to(device) + 0.1

            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)

            # Train Generator
            optimizer_G.zero_grad()
            generated_images = generator(lr_images)
            pred_fake = discriminator(generated_images)
            loss_gan = criterion_gan(pred_fake, real_label)
            loss_pixel = criterion_pixel(generated_images, hr_images)
            loss_G = loss_gan + config.lambda_pixel * loss_pixel
            loss_G.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            pred_real = discriminator(hr_images)
            loss_real = criterion_gan(pred_real, real_label)
            pred_fake_detached = discriminator(generated_images.detach())
            loss_fake = criterion_gan(pred_fake_detached, fake_label)
            loss_D = (loss_real + loss_fake) / 2
            loss_D.backward()
            optimizer_D.step()

            # Log progress and save samples periodically
            if i % config.sample_interval == 0:
                print(f"Epoch [{epoch}/{config.num_epochs}] Batch [{i}] D_loss: {loss_D.item():.4f} G_loss: {loss_G.item():.4f}")
                save_generated_images(lr_images, generated_images, hr_images, sample_dir, epoch, i)

        # Save checkpoint at the end of each epoch
        save_checkpoint(generator, discriminator, optimizer_G, optimizer_D, epoch + 1, checkpoint_dir)

    print("Training completed.")