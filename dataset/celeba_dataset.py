import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF


class CelebARestorationDataset(Dataset):
    def __init__(self, root_dir, split="train", degradation_type="blur"):
        """
        Initialize the CelebA restoration dataset.
        Args:
            root_dir (str): Path to the dataset directory.
            split (str): Dataset split ('train', 'val', or 'test').
            degradation_type (str): Type of degradation ('blur', 'noise', or 'both').
        """
        # Define paths
        self.img_dir = os.path.join(root_dir, "img_align_celeba")
        self.split_file = os.path.join(root_dir, "list_eval_partition.txt")
        self.degradation_type = degradation_type

        # Read split file and filter images based on the split
        self.image_paths = self._get_image_paths(split)

        # Define transformations for high-resolution images
        self.transform_hr = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _get_image_paths(self, split):
        """
        Get image paths based on the specified split.
        Args:
            split (str): Dataset split ('train', 'val', or 'test').
        Returns:
            List of image paths for the specified split.
        """
        split_mapping = {"train": 0, "val": 1, "test": 2}
        if split not in split_mapping:
            raise ValueError(f"Invalid split '{split}'. Must be one of: 'train', 'val', 'test'.")

        with open(self.split_file, "r") as f:
            lines = f.readlines()

        # Filter image paths based on the split value in list_eval_partition.txt
        image_paths = [
            os.path.join(self.img_dir, line.split()[0])
            for line in lines if int(line.split()[1]) == split_mapping[split]
        ]
        return image_paths

    def degrade_image(self, image):
        """Apply degradation to an image."""
        if self.degradation_type == "blur":
            return TF.gaussian_blur(image, kernel_size=[7, 7], sigma=3.0)
        elif self.degradation_type == "noise":
            noise = torch.randn_like(image) * 0.1
            return torch.clamp(image + noise, -1, 1)
        elif self.degradation_type == "both":
            blurred = TF.gaussian_blur(image, kernel_size=[7, 7], sigma=3.0)
            noise = torch.randn_like(blurred) * 0.1
            return torch.clamp(blurred + noise, -1, 1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load high-resolution image
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        
        # Apply transformations to get high-resolution tensor
        hr_img = self.transform_hr(img)
        
        # Generate low-resolution degraded image
        lr_img = self.degrade_image(hr_img)
        
        return lr_img, hr_img