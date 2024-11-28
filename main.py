import torch
from models.generator import Generator
from models.discriminator import Discriminator
from dataset.celeba_dataset import CelebARestorationDataset
from training.train import train_gan
from config.training_config import TrainingConfig

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize models and dataset
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    dataset = CelebARestorationDataset(root_dir="data/celeba", split="train", degradation_type="both")
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=TrainingConfig().batch_size,
                                               shuffle=True,
                                               num_workers=2)

    # Start training process
    train_gan(TrainingConfig(), generator, discriminator, train_loader=train_loader, device=device)

if __name__ == "__main__":
    main()