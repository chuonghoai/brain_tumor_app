import streamlit as st
from PIL import Image
import os

from services.utils import setup_logger
from services.model_loader import get_available_models, load_model
from services.predictor import predict

logger = setup_logger("App")

# ==========================================
# Cấu hình trang
# ==========================================
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Khởi tạo session state
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "EfficientNet-B0"
if "model_ready" not in st.session_state:
    st.session_state.model_ready = False

# Kiểm tra xem file model có tồn tại không trước khi cho phép chạy
def check_model_exists(model_name: str) -> bool:
    if model_name == "EfficientNet-B0":
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "models", "best_efficientnet_b0.pth")
        return os.path.exists(model_path)
    return False

# ==========================================
# Giao diện chính
# ==========================================
st.title("🧠 Brain Tumor Detection using Deep Learning")
st.markdown("---")

# Layout chia cột cho phần điều khiển
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload MRI")
    uploaded_file = st.file_uploader(
        "Chọn ảnh MRI của bạn",
        type=["png", "jpg", "jpeg"]
    )
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            # Cập nhật state nếu có ảnh mới
            if st.session_state.uploaded_image != image:
                st.session_state.uploaded_image = image
                st.session_state.prediction_result = None
        except Exception as e:
            st.error("File tải lên không hợp lệ hoặc bị lỗi. Vui lòng thử ảnh khác.")
            logger.error(f"Lỗi khi đọc file upload: {e}")

with col2:
    st.subheader("Model")
    available_models = get_available_models()
    
    # Cho phép chọn mô hình
    selected_model = st.selectbox(
        "Chọn kiến trúc mô hình",
        available_models,
        index=available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0
    )
    
    # Cập nhật state khi đổi mô hình
    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
        st.session_state.prediction_result = None
        
    st.session_state.model_ready = check_model_exists(st.session_state.selected_model)
    
    if not st.session_state.model_ready:
        st.error(f"Không tìm thấy trọng số cho mô hình {st.session_state.selected_model}. Vui lòng kiểm tra thư mục 'models'.")

st.markdown("---")

# Nút hành động
col_btn1, col_btn2 = st.columns([1, 5])
with col_btn1:
    analyze_btn = st.button(
        "🔍 Analyze", 
        type="primary", 
        use_container_width=True,
        disabled=not st.session_state.model_ready or st.session_state.uploaded_image is None
    )
with col_btn2:
    reset_btn = st.button("🔄 Reset", use_container_width=False)

if reset_btn:
    st.session_state.uploaded_image = None
    st.session_state.prediction_result = None
    st.rerun()

# ==========================================
# Xử lý sự kiện Analyze
# ==========================================
if analyze_btn and st.session_state.uploaded_image is not None:
    try:
        with st.spinner('Đang phân tích ảnh và tạo Grad-CAM...'):
            result = predict(st.session_state.uploaded_image, st.session_state.selected_model)
            st.session_state.prediction_result = result
    except Exception as e:
        st.error("Đã xảy ra lỗi trong quá trình phân tích. Vui lòng kiểm tra log hệ thống.")
        logger.error(f"Lỗi khi chạy predict: {e}")

st.markdown("---")

# ==========================================
# Hiển thị kết quả
# ==========================================
if st.session_state.uploaded_image is not None:
    st.subheader("Visual Analysis")
    
    img_col1, img_col2 = st.columns(2)
    
    with img_col1:
        st.markdown("<h4 style='text-align: center;'>Original MRI</h4>", unsafe_allow_html=True)
        st.image(st.session_state.uploaded_image, use_container_width=True)
        
    with img_col2:
        st.markdown("<h4 style='text-align: center;'>Grad-CAM</h4>", unsafe_allow_html=True)
        if st.session_state.prediction_result is not None:
            heatmap = st.session_state.prediction_result.heatmap
            if heatmap is not None:
                st.image(heatmap, use_container_width=True)
            else:
                st.warning("Không thể sinh ảnh Grad-CAM cho mô hình này.")
        else:
            st.info("Bấm 'Analyze' để tạo ảnh nhiệt Grad-CAM.")

# Hiển thị Metrics nếu đã có kết quả dự đoán
if st.session_state.prediction_result is not None:
    st.markdown("---")
    res = st.session_state.prediction_result
    
    st.subheader("Prediction Report")
    
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    with metrics_col1:
        st.metric(label="Prediction", value=res.prediction)
        
    with metrics_col2:
        st.metric(label="Confidence", value=f"{res.confidence * 100:.2f} %")
        
    with metrics_col3:
        st.metric(label="Inference Time", value=f"{res.inference_time * 1000:.0f} ms")

    st.progress(res.confidence, text="Mức độ tin cậy")

    # Explainability - Top-2 prediction
    with st.expander("Probability Distribution (Top-2)", expanded=True):
        sorted_probs = sorted(res.probabilities.items(), key=lambda item: item[1], reverse=True)
        
        for class_name, prob in sorted_probs:
            col_a, col_b = st.columns([1, 4])
            with col_a:
                st.markdown(f"**{class_name}**")
            with col_b:
                st.progress(prob, text=f"{prob * 100:.2f}%")
