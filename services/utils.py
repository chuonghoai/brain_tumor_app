import logging
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

@dataclass
class PredictionResult:
    """
    Dataclass lưu trữ kết quả dự đoán từ mô hình.
    """
    prediction: str             # Nhãn dự đoán (Tumor / No Tumor)
    confidence: float           # Độ tin cậy (0.0 đến 1.0)
    probabilities: Dict[str, float]  # Xác suất cho từng lớp
    inference_time: float       # Thời gian suy luận (giây)
    heatmap: Optional[np.ndarray] = None # Ảnh numpy mảng của Grad-CAM
    model_name: str = "EfficientNet-B0" # Tên mô hình sử dụng


def setup_logger(name: str = "BrainTumorApp", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger
