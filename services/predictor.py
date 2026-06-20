import time
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn.functional as F
from PIL import Image

from .utils import PredictionResult, setup_logger
from .preprocessing import preprocess_image, preprocess_image_keras
from .model_loader import load_model, MODEL_CONFIG
from .gradcam import TorchGradCAM, KerasGradCAM
import numpy as np

logger = setup_logger("Predictor")

CLASS_NAMES = ["No Tumor", "Tumor"]

def predict(image: Image.Image, model_name: str) -> PredictionResult:
    logger.info("Bắt đầu quá trình suy luận (Prediction started).")
    
    try:
        # Load mô hình
        model = load_model(model_name)
        
        # Đo thời gian
        start_time = time.perf_counter()
        
        filename = MODEL_CONFIG.get(model_name, "")
        is_pytorch = filename.endswith(".pth")
        is_keras = filename.endswith(".keras")
        
        if is_pytorch:
            device = next(model.parameters()).device
            input_tensor = preprocess_image(image).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = F.softmax(output, dim=1)[0].cpu().numpy()
        elif is_keras:
            input_numpy = preprocess_image_keras(image)
            output = model.predict(input_numpy, verbose=0)
            output_flat = output[0]
            if len(output_flat) == 1:
                prob_tumor = float(output_flat[0])
                prob_notumor = 1.0 - prob_tumor
                probabilities = np.array([prob_notumor, prob_tumor])
            else:
                exp_vals = np.exp(output_flat - np.max(output_flat))
                probabilities = exp_vals / np.sum(exp_vals)
        else:
            raise ValueError(f"Không hỗ trợ định dạng cho mô hình {model_name}")
            
        end_time = time.perf_counter()
        inference_time = end_time - start_time
        
        pred_idx = probabilities.argmax()
        prediction_label = CLASS_NAMES[pred_idx]
        confidence = float(probabilities[pred_idx])
        
        prob_dict = {
            CLASS_NAMES[0]: float(probabilities[0]),
            CLASS_NAMES[1]: float(probabilities[1])
        }
        
        logger.info(f"Dự đoán hoàn tất: {prediction_label} ({confidence:.2%})")
        logger.info("Bắt đầu tạo ảnh nhiệt (Heatmap generated).")
        
        # Grad-CAM
        if is_pytorch:
            gradcam = TorchGradCAM(model)
            input_tensor.requires_grad = True
            heatmap = gradcam.generate_heatmap(input_tensor, target_class=pred_idx)
        elif is_keras:
            gradcam = KerasGradCAM(model)
            try:
                heatmap = gradcam.generate_heatmap(input_numpy, target_class=pred_idx)
            except Exception as e:
                logger.warning(f"Lỗi khi tính toán Grad-CAM cho model {model_name}: {e}")
                heatmap = None
            
        overlayed_img = gradcam.overlay_heatmap(image, heatmap, alpha=0.4) if heatmap is not None else None
        
        logger.info("Hoàn tất quy trình suy luận (Prediction finished).")
        
        return PredictionResult(
            prediction=prediction_label,
            confidence=confidence,
            probabilities=prob_dict,
            inference_time=inference_time,
            heatmap=overlayed_img,
            model_name=model_name
        )
        
    except Exception as e:
        logger.error(f"Lỗi trong quá trình suy luận (Prediction failed): {e}")
        raise e
