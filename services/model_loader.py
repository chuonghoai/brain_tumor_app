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

MODEL_CONFIG = {
    "EfficientNet-B0": "best_efficientnet_b0.pth",
    "CNN Custom": "Lan2_best_baseline_cnn_extreme_weights.keras",
    "ResNet50V2": "best_resnet_model.keras",
    "VGG16": "best_vgg16_model.keras"
}

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
def load_model(model_name: str):
    logger.info(f"Bắt đầu tải mô hình: {model_name}")
    
    if model_name not in MODEL_CONFIG:
        logger.error(f"Mô hình {model_name} chưa được cấu hình.")
        raise ValueError(f"Mô hình {model_name} chưa được cấu hình.")
        
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    filename = MODEL_CONFIG[model_name]
    model_path = os.path.join(models_dir, filename)
    
    if not os.path.exists(model_path):
        logger.error(f"Không tìm thấy file mô hình tại: {model_path}")
        raise FileNotFoundError(f"Không tìm thấy file trọng số {model_path}")
        
    if filename.endswith(".pth"):
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
            logger.error(f"Lỗi khi load mô hình PyTorch: {e}")
            raise e
            
    elif filename.endswith(".keras"):
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path, compile=False)
            logger.info(f"Model Keras ({model_name}) loaded thành công.")
            return model
        except Exception as e:
            logger.error(f"Lỗi khi load mô hình Keras: {e}")
            raise e
    else:
        logger.error(f"Định dạng file {filename} của mô hình {model_name} chưa được hỗ trợ.")
        raise ValueError(f"Định dạng file {filename} hiện chưa được hỗ trợ.")

def get_available_models():
    return list(MODEL_CONFIG.keys())
