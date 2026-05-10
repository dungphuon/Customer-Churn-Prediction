import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import shap
import os
from sklearn.pipeline import Pipeline

# ─── Cấu hình trang ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS tùy chỉnh ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border-left: 4px solid #378ADD;
    }
    .metric-card-red   { border-left-color: #E24B4A; }
    .metric-card-green { border-left-color: #1D9E75; }
    .metric-card-amber { border-left-color: #BA7517; }
    .risk-high   { color: #E24B4A; font-weight: 700; }
    .risk-medium { color: #BA7517; font-weight: 700; }
    .risk-low    { color: #1D9E75; font-weight: 700; }
    .info-box {
        background: #E6F1FB;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border: 1px solid #B5D4F4;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a1a2e;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #f0f0f0;
    }
</style>
""", unsafe_allow_html=True)


# ─── Load model (cache) ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    base = os.path.dirname(__file__)
    model_dir = os.path.join(base, "models")
    with open(os.path.join(model_dir, "best_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(model_dir, "feature_meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    scaler = None
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    return model, meta, scaler


@st.cache_resource
def load_explainer(_model):
    """
    TreeExplainer không nhận Pipeline trực tiếp.
    Phải lấy bước 'clf' bên trong ra.
    """
    clf = _model.named_steps['clf']
    return shap.TreeExplainer(clf)


def get_preprocessed(model, X_input):
    """
    Transform X qua tất cả bước trong Pipeline trừ bước cuối (clf),
    để SHAP nhận đúng định dạng.
    """
    preprocessor = Pipeline(model.steps[:-1])
    return preprocessor.transform(X_input)


# ─── Helper functions ─────────────────────────────────────────────────────────
def encode_input(age, gender, tenure, usage_freq, support_calls,
                 payment_delay, sub_type, contract_length, total_spend, last_interaction):
    gender_enc   = 1 if gender == "Female" else 0
    sub_map      = {"Basic": 0, "Standard": 1, "Premium": 2}
    contract_map = {"Monthly": 0, "Quarterly": 1, "Annual": 2}
    return pd.DataFrame([{
        "Age":               float(age),
        "Gender":            gender_enc,
        "Tenure":            int(tenure),
        "Usage Frequency":   int(usage_freq),
        "Support Calls":     int(support_calls),
        "Payment Delay":     float(payment_delay),
        "Subscription Type": sub_map[sub_type],
        "Contract Length":   contract_map[contract_length],
        "Total Spend":       float(total_spend),
        "Last Interaction":  int(last_interaction),
    }])


def risk_label(prob):
    if prob >= 0.7:
        return "Cao", "#E24B4A"
    elif prob >= 0.4:
        return "Trung bình", "#BA7517"
    else:
        return "Thấp", "#1D9E75"


def encode_batch(df_raw):
    df = df_raw.copy()
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map(
            {"Male": 0, "Female": 1, 0: 0, 1: 1}).fillna(0).astype(int)
    if "Subscription Type" in df.columns:
        df["Subscription Type"] = df["Subscription Type"].map(
            {"Basic": 0, "Standard": 1, "Premium": 2, 0: 0, 1: 1, 2: 2}).fillna(0).astype(int)
    if "Contract Length" in df.columns:
        df["Contract Length"] = df["Contract Length"].map(
            {"Monthly": 0, "Quarterly": 1, "Annual": 2, 0: 0, 1: 1, 2: 2}).fillna(0).astype(int)
    return df


# ─── Load và lưu vào session_state ───────────────────────────────────────────
try:
    model, meta, scaler = load_model()
    explainer = load_explainer(model)          # ✅ dùng hàm cache, tự lấy clf bên trong
    st.session_state["model"]     = model
    st.session_state["meta"]      = meta
    st.session_state["scaler"]    = scaler
    st.session_state["explainer"] = explainer
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    st.session_state["load_error"] = str(e)


# ─── UI Trang chủ ─────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">📊 Customer Churn Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Hệ thống dự đoán khả năng rời bỏ dịch vụ của khách hàng</div>',
            unsafe_allow_html=True)

if not MODEL_LOADED:
    st.error(f"❌ Không load được model: {st.session_state.get('load_error', 'unknown')}")
    st.info("Đảm bảo thư mục `models/` chứa: `best_model.pkl`, `feature_meta.pkl`")
    st.stop()

# ─── Metric cards tổng quan ───────────────────────────────────────────────────
st.markdown('<div class="section-header">Thông tin mô hình</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Mô hình", meta.get("best_model_name", "Best Model"))
with col2:
    st.metric("Churn rate (test set)", f"{meta.get('churn_rate_test', 0.341)*100:.1f}%")
with col3:
    st.metric("Số features", len(meta.get("feature_names", [])))
with col4:
    st.metric("Trạng thái", "✅ Sẵn sàng")

st.divider()

# ─── Giới thiệu tính năng ─────────────────────────────────────────────────────
st.markdown('<div class="section-header">Các tính năng</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""
    <div class="metric-card">
        <h4>🔍 Dự đoán đơn lẻ</h4>
        <p style="color:#555;font-size:0.9rem;">
        Nhập thông tin 1 khách hàng → xem xác suất churn ngay lập tức
        kèm biểu đồ SHAP giải thích lý do.
        </p>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div class="metric-card metric-card-green">
        <h4>📂 Dự đoán hàng loạt</h4>
        <p style="color:#555;font-size:0.9rem;">
        Upload file CSV nhiều khách hàng → chạy batch predict →
        lọc theo rủi ro → download kết quả.
        </p>
    </div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown("""
    <div class="metric-card metric-card-amber">
        <h4>📈 Dashboard phân tích</h4>
        <p style="color:#555;font-size:0.9rem;">
        Tổng quan phân phối churn, top khách hàng rủi ro cao,
        SHAP summary toàn bộ dataset.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ─── Giải thích features ──────────────────────────────────────────────────────
st.markdown('<div class="section-header">Ý nghĩa các features đầu vào</div>', unsafe_allow_html=True)

feature_info = pd.DataFrame({
    "Feature": [
        "Age", "Gender", "Tenure", "Usage Frequency",
        "Support Calls", "Payment Delay", "Subscription Type",
        "Contract Length", "Total Spend", "Last Interaction"
    ],
    "Mô tả": [
        "Tuổi khách hàng (18–65)",
        "Giới tính (Male / Female)",
        "Số tháng đã sử dụng dịch vụ",
        "Số lần sử dụng dịch vụ mỗi tháng",
        "Số lần gọi hỗ trợ — nhiều → không hài lòng",
        "Số ngày trễ thanh toán trung bình",
        "Gói đăng ký: Basic / Standard / Premium",
        "Loại hợp đồng: Monthly / Quarterly / Annual",
        "Tổng chi tiêu tích lũy (USD)",
        "Số ngày kể từ lần tương tác gần nhất",
    ],
    "Tín hiệu churn": [
        "Trung tính",
        "Trung tính",
        "Tenure ngắn → dễ rời hơn",
        "Dùng ít → giá trị cảm nhận thấp",
        "Nhiều → không hài lòng ⚠️",
        "Trễ nhiều → rủi ro cao ⚠️",
        "Basic → dễ rời hơn",
        "Monthly → linh hoạt → dễ hủy ⚠️",
        "Chi tiêu thấp → gắn bó kém",
        "Lâu không tương tác → sắp rời ⚠️",
    ]
})

st.dataframe(
    feature_info,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Tín hiệu churn": st.column_config.TextColumn(width="medium"),
    }
)

st.divider()
st.markdown("""
<div class="info-box">
    <b>Hướng dẫn sử dụng:</b><br>
    👈 Chọn tính năng từ thanh điều hướng bên trái để bắt đầu.<br>
    <code>models/best_model.pkl</code> và <code>models/feature_meta.pkl</code>
    phải được đặt đúng thư mục trước khi chạy app.
</div>
""", unsafe_allow_html=True)