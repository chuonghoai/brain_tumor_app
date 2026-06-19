import os
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
from torchvision.models import efficientnet_b0
# pyrefly: ignore [missing-import]
import streamlit as st
from .utils import setup_logger

logger = setup_logger("ModelLoader")

def create_efficientnet_b0():
    model = efficientnet_b0()
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, 2)
    )
    return model

@st.cache_resource
def load_model(model_name: str) -> nn.Module:
    logger.info(f"Bắt đầu tải mô hình: {model_name}")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    
    """
    Chọn model để phân tích ảnh
    """
    if model_name == "EfficientNet-B0":
        model_path = os.path.join(models_dir, "best_efficientnet_b0.pth")
        
        if not os.path.exists(model_path):
            logger.error(f"Không tìm thấy file mô hình tại: {model_path}")
            raise FileNotFoundError(f"Không tìm thấy file trọng số {model_path}")
            
        model = create_efficientnet_b0()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            logger.info("Model loaded thành công.")
            return model
        except Exception as e:
            logger.error(f"Lỗi khi load mô hình: {e}")
            raise e
    else:
        logger.error(f"Mô hình {model_name} chưa được hỗ trợ.")
        raise ValueError(f"Mô hình {model_name} hiện chưa được hỗ trợ.")

def get_available_models():
    """
    Trả về danh sách các mô hình hiện có.
    """
    return ["EfficientNet-B0"]
