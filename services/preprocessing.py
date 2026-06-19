# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from torchvision import transforms
from PIL import Image

def get_preprocessing_transforms() -> transforms.Compose:
    """
    Trả về chuỗi các phép biến đổi (transforms) tiêu chuẩn dùng cho quá trình suy luận (inference).
    Không bao gồm các phép augment dữ liệu ngẫu nhiên.
    
    Quy trình:
    1. Resize ảnh về kích thước 224x224.
    2. Chuyển đổi ảnh thành PyTorch Tensor.
    3. Chuẩn hóa ảnh theo mean và std của ImageNet.
    
    Returns:
        transforms.Compose: Khối biến đổi ảnh.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Thực hiện tiền xử lý cho ảnh MRI đầu vào.
    
    Args:
        image (PIL.Image.Image): Ảnh gốc người dùng tải lên.
        
    Returns:
        torch.Tensor: Tensor ảnh đã được xử lý, thêm chiều batch (1, C, H, W).
    """
    # Đảm bảo ảnh ở hệ màu RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    transform = get_preprocessing_transforms()
    # Thêm batch dimension ở vị trí đầu tiên
    tensor_image = transform(image).unsqueeze(0)
    
    return tensor_image
