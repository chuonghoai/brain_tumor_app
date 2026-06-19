import numpy as np
import cv2
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
import torch.nn.functional as F
from PIL import Image

class BaseGradCAM:
    """
    Lớp trừu tượng cho Grad-CAM. Định nghĩa các phương thức xử lý ảnh nhiệt chung.
    """
    def __init__(self, model):
        self.model = model

    def generate_heatmap(self, input_tensor):
        raise NotImplementedError("Phải triển khai ở lớp con.")

    def overlay_heatmap(self, original_image: Image.Image, heatmap: np.ndarray, alpha: float = 0.4) -> Image.Image:
        """
        Phủ lớp màu (heatmap) lên trên ảnh MRI gốc.

        Args:
            original_image (PIL.Image.Image): Ảnh MRI gốc.
            heatmap (np.ndarray): Ảnh nhiệt 2D có giá trị [0, 1].
            alpha (float): Độ trong suốt của heatmap khi phủ.

        Returns:
            PIL.Image.Image: Ảnh tổng hợp giữa MRI và heatmap.
        """
        # Chuyển ảnh gốc sang numpy array RGB
        img_array = np.array(original_image.convert('RGB'))
        
        # Resize heatmap cho bằng kích thước ảnh gốc
        heatmap_resized = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
        
        # Áp dụng colormap JET từ OpenCV
        heatmap_resized = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        
        # OpenCV dùng BGR, cần chuyển về RGB
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        # Phủ heatmap lên ảnh gốc với hệ số alpha
        superimposed_img = cv2.addWeighted(img_array, 1 - alpha, heatmap_color, alpha, 0)
        
        return Image.fromarray(superimposed_img)


class TorchGradCAM(BaseGradCAM):
    """
    Grad-CAM cho các mô hình PyTorch (vd: EfficientNet-B0).
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module = None):
        super().__init__(model)
        self.model.eval()
        self.target_layer = target_layer
        
        # Tự động tìm lớp Conv2d cuối cùng nếu không được chỉ định cụ thể
        if self.target_layer is None:
            self.target_layer = self._find_last_conv_layer()
            
        self.gradients = None
        self.activations = None
        
        # Đăng ký hook để lấy activations (feature maps) từ target_layer
        self.target_layer.register_forward_hook(self.save_activation)
        # Đăng ký hook để lấy gradients từ target_layer
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def _find_last_conv_layer(self):
        """
        Tìm lớp Conv2d cuối cùng trong toàn bộ mô hình.
        """
        last_conv_layer = None
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv_layer = module
        return last_conv_layer

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """
        Sinh ra ảnh nhiệt Grad-CAM.

        Args:
            input_tensor (torch.Tensor): Tensor đầu vào (1, C, H, W).
            target_class (int): Lớp mục tiêu để tính đạo hàm. Nếu None, chọn lớp có dự đoán cao nhất.

        Returns:
            np.ndarray: Heatmap 2D chuẩn hóa [0, 1].
        """
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        # Tính toán score cho lớp dự đoán
        score = output[0, target_class]
        
        # Backward pass để lấy gradients
        score.backward()
        
        # Lấy gradients và activations từ hooks
        gradients = self.gradients.detach().cpu().numpy()[0] # (C, H, W)
        activations = self.activations.detach().cpu().numpy()[0] # (C, H, W)
        
        # Global Average Pooling trên các kênh của gradients để lấy trọng số alpha
        weights = np.mean(gradients, axis=(1, 2)) # (C,)
        
        # Tính Weighted Feature Maps
        heatmap = np.zeros(activations.shape[1:], dtype=np.float32) # (H, W)
        for i, w in enumerate(weights):
            heatmap += w * activations[i]
            
        # Áp dụng ReLU trên heatmap
        heatmap = np.maximum(heatmap, 0)
        
        # Chuẩn hóa về [0, 1]
        heatmap_max = np.max(heatmap)
        if heatmap_max != 0:
            heatmap /= heatmap_max
            
        return heatmap
