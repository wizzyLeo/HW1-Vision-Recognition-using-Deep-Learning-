from PIL import Image 
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from features.build_features import get_train_transform, get_val_transform, get_test_transform
import os

class TestImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith('.jpg')
        ])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Use Apple's optimized Pillow-SIMD if available (already compiled for M1+ via pip install)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, img_name

BATCH_SIZE = 32
    

def get_train_loader(dataset_path, batch_size = BATCH_SIZE):
    transformer = get_train_transform()
    train_dataset = datasets.ImageFolder(dataset_path, transformer)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return loader

def get_val_loader(dataset_path, batch_size = BATCH_SIZE):
    transformer = get_val_transform()
    val_dataset = datasets.ImageFolder(dataset_path, transformer)
    loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return loader

def get_test_loader(dataset_path, batch_size = BATCH_SIZE):
    transformer = get_test_transform()
    test_dataset = TestImageDataset(dataset_path, transform=transformer)
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return loader