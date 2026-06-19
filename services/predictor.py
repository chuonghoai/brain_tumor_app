import time
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn.functional as F
from PIL import Image

from .utils import PredictionResult, setup_logger
from .preprocessing import preprocess_image
from .model_loader import load_model
from .gradcam import TorchGradCAM

logger = setup_logger("Predictor")

CLASS_NAMES = ["No Tumor", "Tumor"]

def predict(image: Image.Image, model_name: str) -> PredictionResult:
    logger.info("Bắt đầu quá trình suy luận (Prediction started).")
    
    try:
        # Load mô hình
        model = load_model(model_name)
        device = next(model.parameters()).device
        
        # Tiền xử lý
        input_tensor = preprocess_image(image).to(device)
        
        # Đo thời gian
        start_time = time.perf_counter()
        
        # Suy luận
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)[0].cpu().numpy()
            
        end_time = time.perf_counter()
        inference_time = end_time - start_time
        
        # Lấy nhãn và độ tin cậy
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
        gradcam = TorchGradCAM(model)
        input_tensor.requires_grad = True
        heatmap = gradcam.generate_heatmap(input_tensor, target_class=pred_idx)
        overlayed_img = gradcam.overlay_heatmap(image, heatmap, alpha=0.4)
        
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
