from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50


weights = ResNet50_Weights.DEFAULT
default_preprocessor = ResNet50_Weights.DEFAULT.transforms()

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
    # transforms.ToTensor(), 
    default_preprocessor, # 💡 Use meta values here
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3)), 
    
    
])

val_transform = default_preprocessor  # ✅ This is still valid

def get_train_transform():
    return train_transform

def get_val_transform():
    return val_transform

def get_test_transform():
    return default_preprocessor




