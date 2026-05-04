import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io, sys, os

# ─── Increase pandas styler limit for large dataframes
pd.set_option("styler.render.max_elements", 1000000)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from app import encode_batch, risk_label

st.set_page_config(page_title="Dự đoán hàng loạt", page_icon="📂", layout="wide")

st.markdown("""
<style>
    .section-header {
        font-size: 1.05rem; font-weight: 600; color: #1a1a2e;
        margin: 1.2rem 0 0.6rem 0;
        padding-bottom: 0.3rem; border-bottom: 2px solid #f0f0f0;
    }
    .upload-hint {
        background: #f8f9fa; border-radius: 10px;
        padding: 0.8rem 1.2rem; font-size: 0.85rem;
        color: #555; border: 1px dashed #ccc;
        margin-bottom: 1rem;
    }
    .stat-card {
        background: #f8f9fa; border-radius: 10px;
        padding: 1rem 1.2rem; text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.title("📂 Dự đoán hàng loạt")
st.caption("Upload file CSV chứa nhiều khách hàng — app sẽ tự động dự đoán và cho phép lọc, tải kết quả")

if "model" not in st.session_state:
    st.warning("⚠️ Model chưa load. Vui lòng quay lại trang chủ.")
    st.stop()

model    = st.session_state["model"]
meta     = st.session_state["meta"]
FEATURE_NAMES = meta["feature_names"]

# ─── Hướng dẫn format CSV ────────────────────────────────────────────────────
with st.expander("📖 Định dạng file CSV cần upload"):
    st.markdown("""
    File CSV cần có **đúng tên cột** như sau (thứ tự không quan trọng):

    | Cột | Kiểu | Giá trị hợp lệ |
    |-----|------|----------------|
    | Age | số | 18 – 65 |
    | Gender | chữ hoặc số | Male / Female hoặc 0 / 1 |
    | Tenure | số | 1 – 60 |
    | Usage Frequency | số | 1 – 30 |
    | Support Calls | số | 0 – 10 |
    | Payment Delay | số | 0 – 30 |
    | Subscription Type | chữ hoặc số | Basic / Standard / Premium hoặc 0 / 1 / 2 |
    | Contract Length | chữ hoặc số | Monthly / Quarterly / Annual hoặc 0 / 1 / 2 |
    | Total Spend | số | 100 – 1000 |
    | Last Interaction | số | 1 – 30 |

    Không cần cột `churn_consensus` hay `risk_score`.
    """)

    # Tạo file mẫu cho user tải
    sample_data = pd.DataFrame({
        "Age": [35, 52, 28, 45],
        "Gender": ["Male", "Female", "Male", "Female"],
        "Tenure": [12, 48, 3, 36],
        "Usage Frequency": [5, 20, 2, 15],
        "Support Calls": [7, 1, 8, 3],
        "Payment Delay": [25.0, 2.0, 28.0, 8.0],
        "Subscription Type": ["Basic", "Premium", "Basic", "Standard"],
        "Contract Length": ["Monthly", "Annual", "Monthly", "Quarterly"],
        "Total Spend": [200.0, 850.0, 150.0, 500.0],
        "Last Interaction": [28, 3, 29, 12],
    })
    csv_sample = sample_data.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Tải file CSV mẫu",
        data=csv_sample,
        file_name="sample_input.csv",
        mime="text/csv",
    )

# ─── Upload ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Upload file CSV</div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Chọn file CSV",
    type=["csv"],
    help="File CSV với 10 cột features như hướng dẫn bên trên"
)

if uploaded is None:
    st.markdown("""
    <div class="upload-hint">
        👆 Upload file CSV để bắt đầu dự đoán hàng loạt.<br>
        Bạn có thể tải file mẫu ở phần hướng dẫn bên trên để xem định dạng đúng.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─── Đọc & validate ───────────────────────────────────────────────────────────
try:
    df_raw = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"❌ Lỗi đọc CSV: {e}")
    st.stop()

# Kiểm tra cột
missing_cols = [c for c in FEATURE_NAMES if c not in df_raw.columns]
if missing_cols:
    st.error(f"❌ Thiếu cột: {missing_cols}")
    st.stop()

# Bỏ các cột thừa (risk_score, churn_consensus nếu có)
df_feat = df_raw[FEATURE_NAMES].copy()

st.success(f"✅ Đọc thành công: **{len(df_feat):,} khách hàng**, {df_feat.shape[1]} features")

with st.expander("Xem 5 dòng đầu dữ liệu"):
    st.dataframe(df_raw.head(), use_container_width=True)

# ─── Predict ──────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Đang dự đoán...</div>', unsafe_allow_html=True)

with st.spinner("Đang chạy mô hình..."):
    df_enc = encode_batch(df_feat)
    df_enc = df_enc[FEATURE_NAMES]

    probs = model.predict_proba(df_enc)[:, 1]
    preds = (probs >= 0.5).astype(int)
    labels = [risk_label(p)[0] for p in probs]

df_result = df_raw.copy()
df_result["churn_probability"] = probs.round(4)
df_result["churn_prediction"]  = preds
df_result["risk_level"]        = labels

# ─── Thống kê tổng quan ───────────────────────────────────────────────────────
st.markdown('<div class="section-header">Tổng quan kết quả</div>', unsafe_allow_html=True)

n_high   = (np.array(labels) == "Cao").sum()
n_mid    = (np.array(labels) == "Trung bình").sum()
n_low    = (np.array(labels) == "Thấp").sum()
n_churn  = preds.sum()
avg_prob = probs.mean()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Tổng khách hàng",    f"{len(df_result):,}")
c2.metric("Dự đoán churn",      f"{n_churn:,}",  f"{n_churn/len(df_result)*100:.1f}%")
c3.metric("Rủi ro Cao",         f"{n_high:,}",   delta=f"{n_high/len(df_result)*100:.1f}%",
          delta_color="inverse")
c4.metric("Rủi ro Trung bình",  f"{n_mid:,}")
c5.metric("Xác suất TB",        f"{avg_prob*100:.1f}%")

# ─── Biểu đồ phân phối ────────────────────────────────────────────────────────
col_hist, col_pie = st.columns(2)

with col_hist:
    fig_hist = px.histogram(
        df_result, x="churn_probability",
        nbins=40, color_discrete_sequence=["#378ADD"],
        labels={"churn_probability": "Xác suất churn"},
        title="Phân phối xác suất churn",
    )
    fig_hist.add_vline(x=0.5, line_dash="dash", line_color="#E24B4A",
                       annotation_text="Ngưỡng 50%", annotation_position="top right")
    fig_hist.update_layout(
        height=300, margin=dict(t=50, b=30, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with col_pie:
    fig_pie = go.Figure(go.Pie(
        labels=["Rủi ro Thấp", "Rủi ro Trung bình", "Rủi ro Cao"],
        values=[n_low, n_mid, n_high],
        marker_colors=["#1D9E75", "#BA7517", "#E24B4A"],
        hole=0.45,
        textinfo="label+percent",
        hovertemplate="%{label}: %{value:,} khách hàng<extra></extra>",
    ))
    fig_pie.update_layout(
        title="Phân loại mức rủi ro",
        height=300, margin=dict(t=50, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# ─── Bảng kết quả + filter ───────────────────────────────────────────────────
st.markdown('<div class="section-header">Bảng kết quả chi tiết</div>', unsafe_allow_html=True)

col_filter1, col_filter2, col_filter3 = st.columns([1, 1, 2])
with col_filter1:
    filter_risk = st.multiselect(
        "Lọc theo mức rủi ro",
        options=["Cao", "Trung bình", "Thấp"],
        default=["Cao", "Trung bình", "Thấp"],
    )
with col_filter2:
    filter_pred = st.selectbox(
        "Lọc theo dự đoán",
        options=["Tất cả", "Chỉ churn (1)", "Chỉ không churn (0)"],
    )
with col_filter3:
    prob_range = st.slider(
        "Lọc theo khoảng xác suất",
        min_value=0.0, max_value=1.0,
        value=(0.0, 1.0), step=0.01,
        format="%.2f",
    )

# Áp dụng filter
mask = (
    df_result["risk_level"].isin(filter_risk) &
    df_result["churn_probability"].between(*prob_range)
)
if filter_pred == "Chỉ churn (1)":
    mask &= df_result["churn_prediction"] == 1
elif filter_pred == "Chỉ không churn (0)":
    mask &= df_result["churn_prediction"] == 0

df_filtered = df_result[mask].sort_values("churn_probability", ascending=False)

st.caption(f"Hiển thị {len(df_filtered):,} / {len(df_result):,} khách hàng")

# Highlight màu theo risk
def highlight_risk(val):
    colors_map = {"Cao": "background-color:#FCEBEB;color:#A32D2D;font-weight:bold",
                  "Trung bình": "background-color:#FAEEDA;color:#854F0B;font-weight:bold",
                  "Thấp": "background-color:#E1F5EE;color:#0F6E56;font-weight:bold"}
    return colors_map.get(val, "")

def highlight_prob(val):
    try:
        v = float(val)
        if v >= 0.7:   return "background-color:#FCEBEB;color:#A32D2D"
        elif v >= 0.4: return "background-color:#FAEEDA;color:#854F0B"
        else:          return "background-color:#E1F5EE;color:#0F6E56"
    except:
        return ""

styled = (df_filtered
          .style
          .map(highlight_risk,    subset=["risk_level"])
          .map(highlight_prob,    subset=["churn_probability"])
          .format({"churn_probability": "{:.4f}"}))

st.dataframe(styled, use_container_width=True, height=400)

# ─── Top rủi ro cao nhất ─────────────────────────────────────────────────────
st.markdown('<div class="section-header">Top 15 khách hàng rủi ro cao nhất</div>',
            unsafe_allow_html=True)

top15 = df_result.nlargest(15, "churn_probability").reset_index()
top15["rank"] = range(1, len(top15) + 1)

fig_top = go.Figure(go.Bar(
    x=top15["churn_probability"],
    y=[f"#{r} (idx {i})" for r, i in zip(top15["rank"], top15["index"])],
    orientation="h",
    marker_color=[
        "#E24B4A" if p >= 0.7 else "#BA7517" if p >= 0.4 else "#1D9E75"
        for p in top15["churn_probability"]
    ],
    text=[f"{p*100:.1f}%" for p in top15["churn_probability"]],
    textposition="outside",
))
fig_top.add_vline(x=0.5, line_dash="dash", line_color="#555",
                  annotation_text="50%", annotation_position="top")
fig_top.update_layout(
    height=430,
    xaxis=dict(range=[0, 1.15], title="Xác suất churn"),
    yaxis=dict(autorange="reversed"),
    margin=dict(l=130, r=80, t=30, b=40),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_top, use_container_width=True)

# ─── Download kết quả ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Tải kết quả</div>', unsafe_allow_html=True)

col_dl1, col_dl2 = st.columns(2)

with col_dl1:
    csv_all = df_result.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Tải toàn bộ kết quả (CSV)",
        data=csv_all,
        file_name="churn_predictions_all.csv",
        mime="text/csv",
        use_container_width=True,
    )

with col_dl2:
    csv_filtered = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Tải kết quả đã lọc (CSV)",
        data=csv_filtered,
        file_name="churn_predictions_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )

# Gợi ý Dashboard
st.info("💡 Chuyển sang trang **Dashboard phân tích** để xem SHAP summary "
        "và phân tích sâu theo segment.")
