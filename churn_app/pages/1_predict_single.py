import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import shap
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ─── Import helpers từ app.py ─────────────────────────────────────────────────
from app import encode_input, risk_label

st.set_page_config(page_title="Dự đoán đơn lẻ", page_icon="🔍", layout="wide")

st.markdown("""
<style>
    .result-box {
        border-radius: 14px;
        padding: 1.4rem 1.8rem;
        margin: 1rem 0;
        text-align: center;
    }
    .result-high   { background: #FCEBEB; border: 2px solid #E24B4A; }
    .result-medium { background: #FAEEDA; border: 2px solid #BA7517; }
    .result-low    { background: #E1F5EE; border: 2px solid #1D9E75; }
    .result-prob   { font-size: 3rem; font-weight: 800; margin: 0.3rem 0; }
    .result-label  { font-size: 1.2rem; font-weight: 600; }
    .section-header {
        font-size: 1.05rem; font-weight: 600;
        color: #1a1a2e;
        margin: 1.2rem 0 0.6rem 0;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #f0f0f0;
    }
    .shap-explain {
        font-size: 0.85rem;
        color: #666;
        background: #f8f9fa;
        border-radius: 8px;
        padding: 0.7rem 1rem;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔍 Dự đoán đơn lẻ")
st.caption("Nhập thông tin một khách hàng để xem xác suất và lý do có thể rời dịch vụ")

# ─── Kiểm tra model đã load ───────────────────────────────────────────────────
if "model" not in st.session_state:
    st.warning("⚠️ Model chưa được load. Vui lòng quay lại trang chủ trước.")
    st.stop()

model    = st.session_state["model"]
explainer = st.session_state["explainer"]
meta     = st.session_state["meta"]
FEATURE_NAMES = meta["feature_names"]

# ─── Form nhập thông tin ──────────────────────────────────────────────────────
st.markdown('<div class="section-header">Thông tin khách hàng</div>', unsafe_allow_html=True)

with st.form("predict_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Thông tin cá nhân**")
        age    = st.number_input("Tuổi", min_value=18, max_value=65, value=35, step=1)
        gender = st.selectbox("Giới tính", ["Male", "Female"])
        tenure = st.number_input("Tenure (tháng)", min_value=1, max_value=60, value=24, step=1,
                                 help="Số tháng đã sử dụng dịch vụ")

    with col2:
        st.markdown("**Hành vi sử dụng**")
        usage_freq     = st.number_input("Usage Frequency", min_value=1, max_value=30, value=10,
                                         help="Số lần dùng dịch vụ mỗi tháng")
        support_calls  = st.number_input("Support Calls", min_value=0, max_value=10, value=2,
                                         help="Số lần gọi hỗ trợ — nhiều = không hài lòng")
        last_interaction = st.number_input("Last Interaction (ngày)", min_value=1, max_value=30, value=10,
                                           help="Số ngày từ lần tương tác gần nhất")

    with col3:
        st.markdown("**Tài chính & Gói dịch vụ**")
        payment_delay   = st.number_input("Payment Delay (ngày)", min_value=0.0, max_value=30.0,
                                          value=5.0, step=0.5,
                                          help="Số ngày trễ thanh toán trung bình")
        total_spend     = st.number_input("Total Spend (USD)", min_value=100.0, max_value=1000.0,
                                          value=400.0, step=10.0)
        sub_type        = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
        contract_length = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])

    submitted = st.form_submit_button("🔮 Dự đoán ngay", use_container_width=True, type="primary")

# ─── Xử lý kết quả ───────────────────────────────────────────────────────────
if submitted:
    X_input = encode_input(age, gender, tenure, usage_freq, support_calls,
                           payment_delay, sub_type, contract_length, total_spend, last_interaction)
    X_input = X_input[FEATURE_NAMES]

    prob   = model.predict_proba(X_input)[0][1]
    pred   = int(prob >= 0.5)
    label, color = risk_label(prob)

    # ── Kết quả chính ──
    st.markdown('<div class="section-header">Kết quả dự đoán</div>', unsafe_allow_html=True)

    col_res, col_gauge = st.columns([1, 1.4])

    with col_res:
        css_class = {"Cao": "result-high", "Trung bình": "result-medium", "Thấp": "result-low"}[label]
        icon      = {"Cao": "🔴", "Trung bình": "🟡", "Thấp": "🟢"}[label]

        st.markdown(f"""
        <div class="result-box {css_class}">
            <div class="result-prob" style="color:{color};">{prob*100:.1f}%</div>
            <div class="result-label" style="color:{color};">{icon} Rủi ro {label}</div>
            <div style="color:#888;font-size:0.85rem;margin-top:0.5rem;">
                Ngưỡng quyết định: 50% | Baseline: 34.1%
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Khuyến nghị hành động
        st.markdown("**Khuyến nghị:**")
        if label == "Cao":
            st.error("⚡ Ưu tiên liên hệ ngay — khách hàng rủi ro rất cao. "
                     "Cân nhắc ưu đãi giữ chân hoặc cuộc gọi từ account manager.")
        elif label == "Trung bình":
            st.warning("📧 Theo dõi chặt hơn — gửi email ưu đãi, khảo sát mức hài lòng "
                       "trong 2 tuần tới.")
        else:
            st.success("✅ Rủi ro thấp — duy trì chương trình loyalty hiện tại.")

    with col_gauge:
        # Gauge chart
        bar_color = {"Cao": "#E24B4A", "Trung bình": "#BA7517", "Thấp": "#1D9E75"}[label]
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(prob * 100, 1),
            number={"suffix": "%", "font": {"size": 40}},
            title={"text": "Xác suất Churn", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#aaa"},
                "bar":  {"color": bar_color, "thickness": 0.25},
                "bgcolor": "white",
                "steps": [
                    {"range": [0, 40],   "color": "#E1F5EE"},
                    {"range": [40, 70],  "color": "#FAEEDA"},
                    {"range": [70, 100], "color": "#FCEBEB"},
                ],
                "threshold": {
                    "line": {"color": "#555", "width": 2},
                    "thickness": 0.75,
                    "value": 50,
                },
            }
        ))
        fig_gauge.update_layout(
            height=280, margin=dict(t=40, b=10, l=30, r=30),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ── SHAP Waterfall ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Giải thích dự đoán (SHAP)</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="shap-explain">
        <b>Đọc biểu đồ:</b>
        Cột <span style="color:#E24B4A;font-weight:600;">đỏ</span> = feature này <b>tăng</b> xác suất churn.
        Cột <span style="color:#378ADD;font-weight:600;">xanh</span> = feature này <b>giảm</b> xác suất churn.
        Độ dài cột = mức độ ảnh hưởng.
    </div>
    """, unsafe_allow_html=True)

    # Tính SHAP
    shap_vals = explainer.shap_values(X_input)
    # Xử lý cả 2 dạng API (cũ/mới)
    if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        sv = shap_vals[0, :, 1]
    elif isinstance(shap_vals, list):
        sv = np.array(shap_vals[1][0])
    else:
        sv = shap_vals[0]

    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = float(base_val[1])
    else:
        base_val = float(base_val)

    # Sắp xếp theo |SHAP| giảm dần, lấy top 10
    idx = np.argsort(np.abs(sv))[::-1]
    sv_sorted  = sv[idx]
    fn_sorted  = [FEATURE_NAMES[i] for i in idx]
    val_sorted = [X_input.iloc[0, i] for i in idx]

    # Label trực quan hơn cho từng feature
    label_map = {
        "Gender": {0: "Male", 1: "Female"},
        "Subscription Type": {0: "Basic", 1: "Standard", 2: "Premium"},
        "Contract Length": {0: "Monthly", 1: "Quarterly", 2: "Annual"},
    }
    y_labels = []
    for fn, raw_val in zip(fn_sorted, val_sorted):
        if fn in label_map:
            display = label_map[fn].get(int(raw_val), str(raw_val))
        elif fn in ["Payment Delay", "Total Spend"]:
            display = f"{raw_val:.1f}"
        else:
            display = str(int(raw_val))
        y_labels.append(f"{fn} = {display}")

    colors  = ["#E24B4A" if v > 0 else "#378ADD" for v in sv_sorted]

    fig_shap = go.Figure()
    fig_shap.add_trace(go.Bar(
        x=sv_sorted[::-1],
        y=y_labels[::-1],
        orientation="h",
        marker_color=colors[::-1],
        text=[f"{'+' if v > 0 else ''}{v:.4f}" for v in sv_sorted[::-1]],
        textposition="outside",
        cliponaxis=False,
    ))

    max_abs = max(abs(sv_sorted)) if len(sv_sorted) > 0 else 0.1
    fig_shap.add_vline(x=0, line_width=1.5, line_color="#aaa")
    fig_shap.update_layout(
        title=f"SHAP Values (base value = {base_val:.3f})",
        xaxis_title="SHAP Value (tác động đến xác suất churn)",
        xaxis=dict(range=[-(max_abs * 1.4), max_abs * 1.4]),
        height=420,
        margin=dict(l=220, r=80, t=50, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(showgrid=False),
        xaxis_showgrid=True,
    )
    st.plotly_chart(fig_shap, use_container_width=True)

    # ── Bảng chi tiết input ───────────────────────────────────────────────────
    with st.expander("📋 Xem chi tiết thông tin đã nhập"):
        display_df = pd.DataFrame({
            "Feature": FEATURE_NAMES,
            "Giá trị": [
                age, "Female" if gender == "Female" else "Male",
                tenure, usage_freq, support_calls, payment_delay,
                sub_type, contract_length, total_spend, last_interaction
            ],
            "Encoded": X_input.iloc[0].values.tolist(),
        })
        st.dataframe(display_df, hide_index=True, use_container_width=True)
