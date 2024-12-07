import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.utils import save_image

# Define Generator Network (same as in training)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # Residual Blocks (use 8 residual blocks as in training)
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(4)]  # Ensure there are 8 residual blocks
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.residual_blocks(x)
        x = self.decoder(x)
        return x

# Define Residual Block (same as in training)
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return out

def load_generator(checkpoint_path, device):
    # Initialize generator model
    generator = Generator().to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load generator state dict (weights)
    generator.load_state_dict(checkpoint['generator_state_dict'])

    # Set to evaluation mode
    generator.eval()

    return generator

def degrade_image(image):
    """Apply degradation (blur + noise) to an image."""
    blurred = TF.gaussian_blur(image, kernel_size=[7, 7], sigma=3.0)
    noise = torch.randn_like(blurred) * 0.1
    return torch.clamp(blurred + noise, -1, 1)

def infer(generator, noisy_image):
    # Ensure input is on the same device as the model
    noisy_image = noisy_image.to(next(generator.parameters()).device)

    # Run inference through the generator
    with torch.no_grad():
        restored_image = generator(noisy_image.unsqueeze(0))  # Add batch dimension
    
    return restored_image.squeeze(0)  # Remove batch dimension

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load pre-trained generator model from checkpoint
    checkpoint_path = 'celeba_gan_results/checkpoints/checkpoint_epoch_40.pth' 
    generator = load_generator(checkpoint_path, device)

    # Load and preprocess a test image (assuming it's stored locally)
    img_path = 'image.png' 
    img = Image.open(img_path).convert('RGB')

    transform_hr = transforms.Compose([
        transforms.CenterCrop(178),  # CelebA face region
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    hr_img = transform_hr(img)  # High-quality image transformation

    # Degrade the image (simulate a noisy or blurred input)
    lr_img = degrade_image(hr_img)

    # Run inference using the loaded generator model
    restored_img = infer(generator, lr_img)

    # Optionally save or display restored image using torchvision.utils.save_image()
    save_image(restored_img, 'restored_image.png', normalize=True)

if __name__ == "__main__":
    main()