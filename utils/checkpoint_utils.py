import os
import torch

def save_checkpoint(generator, discriminator, optimizer_G, optimizer_D, epoch, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
    checkpoint_data = {
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "optimizer_G_state_dict": optimizer_G.state_dict(),
        "optimizer_D_state_dict": optimizer_D.state_dict(),
        "epoch": epoch,
    }
    torch.save(checkpoint_data, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")