import streamlit as st
from PIL import Image
import os

from services.utils import setup_logger
from services.model_loader import get_available_models, load_model
from services.predictor import predict

logger = setup_logger("App")
st.set_page_config(
    page_title="Phát Hiện Khối U Não",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Init state
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "EfficientNet-B0"
if "model_ready" not in st.session_state:
    st.session_state.model_ready = False

# Kiểm tra file model đã tồn tại trong session chưa
def check_model_exists(model_name: str) -> bool:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if model_name == "EfficientNet-B0":
        model_path = os.path.join(base_dir, "models", "best_efficientnet_b0.pth")
        return os.path.exists(model_path)
    elif model_name == "CNN Custom":
        model_path = os.path.join(base_dir, "models", "Lan2_best_baseline_cnn_extreme_weights.keras")
        return os.path.exists(model_path)
    return False
st.session_state.model_ready = check_model_exists(st.session_state.selected_model)

def reset_session():
    st.session_state.uploaded_image = None
    st.session_state.prediction_result = None

# Header
header_left, header_right = st.columns([8, 1])
with header_left:
    st.markdown("<h1 style='text-align: center;'>PHÁT HIỆN KHỐI U NÃO TRÊN ẢNH MRI</h1>", unsafe_allow_html=True)
with header_right:
    if st.session_state.uploaded_image is not None:
        if st.button("Làm mới", use_container_width=True):
            reset_session()
            st.rerun()

st.markdown("---")

# Main UI
col_left, col_right = st.columns(2)

# Vùng tải ảnh lên
with col_left:
    st.subheader("Tải ảnh MRI")

    if st.session_state.uploaded_image is None:
        # Chưa có ảnh
        uploaded_file = st.file_uploader(
            "Chọn ảnh MRI của bạn",
            type=["png", "jpg", "jpeg"]
        )
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.session_state.uploaded_image = image
                st.session_state.prediction_result = None
                st.rerun()
            except Exception as e:
                st.error("Tập tin tải lên không hợp lệ hoặc bị lỗi. Vui lòng thử ảnh khác.")
                logger.error(f"Lỗi khi đọc tập tin tải lên: {e}")
    else:
        # Đã có ảnh
        st.image(st.session_state.uploaded_image, use_container_width=True)
        if st.button("Xóa ảnh", use_container_width=True):
            reset_session()
            st.rerun()

# Nút phân tích ảnh - Kết quả phân tích
with col_right:
    if st.session_state.prediction_result is None:
        # Lúc ban đầu chưa bấm phân tích
        st.subheader("Điều khiển")
        
        available_models = get_available_models()
        selected_model = st.selectbox(
            "Chọn mô hình",
            available_models,
            index=available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0
        )
        
        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model
            st.session_state.model_ready = check_model_exists(selected_model)
            st.rerun()
            
        if not st.session_state.model_ready:
            st.error(f"Không tìm thấy trọng số cho mô hình {st.session_state.selected_model}.")
            
        analyze_clicked = st.button(
            "Phân tích ảnh", 
            type="primary", 
            use_container_width=True,
            disabled=not st.session_state.model_ready or st.session_state.uploaded_image is None
        )
        
        if analyze_clicked and st.session_state.uploaded_image is not None:
            try:
                with st.spinner('Đang xử lý ảnh...'):
                    result = predict(st.session_state.uploaded_image, st.session_state.selected_model)
                    st.session_state.prediction_result = result
                st.rerun()
            except Exception as e:
                st.error("Đã xảy ra lỗi trong quá trình phân tích. Vui lòng thử lại hoặc kiểm tra hệ thống.")
                logger.error(f"Lỗi khi xử lý predict: {e}")
                
    else:
        # Model đã phân tích xong ảnh
        st.subheader("Báo cáo Chẩn đoán")
        res = st.session_state.prediction_result
        
        if res.heatmap is not None:
            st.image(res.heatmap, use_container_width=True)
        else:
            st.warning("Không thể sinh ảnh bản đồ nhiệt cho mô hình này.")
            
        st.markdown("---")
        
        # Report
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        with metrics_col1:
            st.metric(label="Kết luận", value=res.prediction)
        with metrics_col2:
            st.metric(label="Độ tin cậy", value=f"{res.confidence * 100:.2f}%")
        with metrics_col3:
            st.metric(label="Thời gian xử lý", value=f"{res.inference_time * 1000:.0f} ms")
            
        st.progress(res.confidence, text="Mức độ tin cậy")
        
        with st.expander("Phân phối xác suất", expanded=True):
            sorted_probs = sorted(res.probabilities.items(), key=lambda item: item[1], reverse=True)
            for class_name, prob in sorted_probs:
                col_name, col_prog = st.columns([1, 4])
                with col_name:
                    st.markdown(f"**{class_name}**")
                with col_prog:
                    st.progress(prob, text=f"{prob * 100:.2f}%")
