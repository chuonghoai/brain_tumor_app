# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from torchvision import transforms
from PIL import Image
import numpy as np

def get_preprocessing_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def preprocess_image(image: Image.Image) -> torch.Tensor:
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    transform = get_preprocessing_transforms()
    tensor_image = transform(image).unsqueeze(0)
    return tensor_image

def preprocess_image_keras(image: Image.Image) -> np.ndarray:
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    image = image.resize((224, 224))
    img_array = np.array(image).astype(np.float32)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array
