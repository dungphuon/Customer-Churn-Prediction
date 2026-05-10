import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from app import encode_batch, risk_label

st.set_page_config(page_title="Dashboard Phân Tích", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .section-header {
        font-size: 1.05rem; font-weight: 600; color: #1a1a2e;
        margin: 1.4rem 0 0.7rem 0;
        padding-bottom: 0.3rem; border-bottom: 2px solid #f0f0f0;
    }
    .insight-box {
        background: #E6F1FB; border-radius: 10px;
        padding: 0.8rem 1.1rem; border-left: 4px solid #378ADD;
        font-size: 0.88rem; color: #185FA5; margin-bottom: 0.8rem;
    }
    .warn-box {
        background: #FAEEDA; border-radius: 10px;
        padding: 0.8rem 1.1rem; border-left: 4px solid #BA7517;
        font-size: 0.88rem; color: #854F0B; margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 Dashboard phân tích toàn diện")
st.caption("Tổng quan churn, tầm quan trọng features, và phân tích theo từng phân khúc khách hàng")

if "model" not in st.session_state:
    st.warning("⚠️ Model chưa load. Vui lòng quay lại trang chủ.")
    st.stop()

model     = st.session_state["model"]
explainer = st.session_state["explainer"]
meta      = st.session_state["meta"]
FEATURE_NAMES = meta["feature_names"]

# ─── Nguồn dữ liệu ────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Nguồn dữ liệu phân tích</div>', unsafe_allow_html=True)

data_source = st.radio(
    "Chọn dữ liệu để phân tích:",
    ["📁 Upload CSV", "🎲 Dùng dữ liệu mẫu (demo)"],
    horizontal=True,
)

df_enc = None

if data_source == "📁 Upload CSV":
    uploaded = st.file_uploader("Upload CSV (cùng định dạng như trang Batch)", type=["csv"])
    if uploaded:
        df_raw = pd.read_csv(uploaded)
        missing = [c for c in FEATURE_NAMES if c not in df_raw.columns]
        if missing:
            st.error(f"❌ Thiếu cột: {missing}")
            st.stop()
        df_enc = encode_batch(df_raw[FEATURE_NAMES].copy())[FEATURE_NAMES]
        st.success(f"✅ {len(df_enc):,} khách hàng")
    else:
        st.info("Upload file CSV để bắt đầu phân tích.")
        st.stop()

else:
    # Tạo dữ liệu demo ngẫu nhiên
    np.random.seed(42)
    n = 500
    df_enc = pd.DataFrame({
        "Age":               np.random.randint(18, 66, n).astype(float),
        "Gender":            np.random.choice([0, 1], n),
        "Tenure":            np.random.randint(1, 61, n),
        "Usage Frequency":   np.random.randint(1, 30, n),
        "Support Calls":     np.random.randint(0, 10, n),
        "Payment Delay":     np.random.uniform(0, 30, n),
        "Subscription Type": np.random.choice([0, 1, 2], n),
        "Contract Length":   np.random.choice([0, 1, 2], n),
        "Total Spend":       np.random.uniform(100, 1000, n),
        "Last Interaction":  np.random.randint(1, 30, n),
    })
    st.info(f"Đang dùng {n} mẫu dữ liệu demo ngẫu nhiên.")

# ─── Predict toàn bộ ─────────────────────────────────────────────────────────
with st.spinner("Đang tính toán..."):
    df_enc = df_enc[FEATURE_NAMES]
    probs  = model.predict_proba(df_enc)[:, 1]
    preds  = (probs >= 0.5).astype(int)
    labels_arr = [risk_label(p)[0] for p in probs]

    df_full = df_enc.copy()
    df_full["churn_probability"] = probs
    df_full["churn_prediction"]  = preds
    df_full["risk_level"]        = labels_arr

n_total  = len(df_full)
n_churn  = preds.sum()
n_high   = (np.array(labels_arr) == "Cao").sum()
n_mid    = (np.array(labels_arr) == "Trung bình").sum()
n_low    = (np.array(labels_arr) == "Thấp").sum()

# ─── KPI row ──────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Tổng quan</div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Tổng mẫu",          f"{n_total:,}")
c2.metric("Dự đoán churn",     f"{n_churn:,}", f"{n_churn/n_total*100:.1f}%")
c3.metric("Rủi ro Cao",        f"{n_high:,}",  f"{n_high/n_total*100:.1f}%", delta_color="inverse")
c4.metric("Rủi ro Trung bình", f"{n_mid:,}",   f"{n_mid/n_total*100:.1f}%",  delta_color="off")
c5.metric("Xác suất TB",       f"{probs.mean()*100:.1f}%")

# ─── Phân phối churn + Risk donut ─────────────────────────────────────────────
col_h, col_d = st.columns(2)

with col_h:
    fig_hist = px.histogram(
        df_full, x="churn_probability", nbins=50,
        color="risk_level",
        color_discrete_map={"Cao": "#E24B4A", "Trung bình": "#BA7517", "Thấp": "#1D9E75"},
        labels={"churn_probability": "Xác suất churn", "risk_level": "Mức rủi ro"},
        title="Phân phối xác suất churn theo mức rủi ro",
        barmode="overlay",
        opacity=0.75,
    )
    fig_hist.add_vline(x=0.5, line_dash="dash", line_color="#333",
                       annotation_text="Ngưỡng 50%")
    fig_hist.update_layout(
        height=320, margin=dict(t=50, b=30, l=30, r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with col_d:
    fig_donut = go.Figure(go.Pie(
        labels=["Thấp", "Trung bình", "Cao"],
        values=[n_low, n_mid, n_high],
        marker_colors=["#1D9E75", "#BA7517", "#E24B4A"],
        hole=0.55,
        textinfo="label+percent+value",
        hovertemplate="%{label}: %{value:,} KH (%{percent})<extra></extra>",
    ))
    fig_donut.update_layout(
        title="Tỷ lệ phân loại rủi ro",
        height=320, margin=dict(t=50, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(text=f"{n_churn/n_total*100:.1f}%<br>churn",
                          font_size=16, showarrow=False)],
        showlegend=False,
    )
    st.plotly_chart(fig_donut, use_container_width=True)

# ─── SHAP Summary (Mean |SHAP|) ───────────────────────────────────────────────
st.markdown('<div class="section-header">Tầm quan trọng Features — SHAP Summary</div>',
            unsafe_allow_html=True)

st.markdown("""
<div class="insight-box">
    <b>SHAP Mean |SHAP|:</b> Trung bình giá trị tuyệt đối của SHAP trên toàn bộ dataset.
    Feature nào có giá trị cao hơn → ảnh hưởng đến dự đoán nhiều hơn.
    Khác với feature importance của mô hình ở chỗ: SHAP đo tác động thực tế lên <i>từng dự đoán</i>.
</div>
""", unsafe_allow_html=True)

MAX_SHAP_SAMPLES = 300
sample_idx = np.random.choice(len(df_enc), min(MAX_SHAP_SAMPLES, len(df_enc)), replace=False)
df_shap_sample = df_enc.iloc[sample_idx].reset_index(drop=True)

with st.spinner(f"Đang tính SHAP trên {len(df_shap_sample)} mẫu..."):
    # ✅ Sửa: transform qua preprocessor trước
    from sklearn.pipeline import Pipeline as _Pipeline
    _preprocessor = _Pipeline(model.steps[:-1])
    df_shap_transformed = _preprocessor.transform(df_shap_sample)
    shap_vals = explainer.shap_values(df_shap_transformed)
    if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        sv_all = shap_vals[:, :, 1]
    elif isinstance(shap_vals, list):
        sv_all = np.array(shap_vals[1])
    else:
        sv_all = shap_vals

mean_abs_shap = np.abs(sv_all).mean(axis=0)
df_shap_imp = pd.DataFrame({
    "Feature":    FEATURE_NAMES,
    "Mean |SHAP|": mean_abs_shap
}).sort_values("Mean |SHAP|", ascending=True)

col_shap, col_fi = st.columns(2)

with col_shap:
    fig_shap = go.Figure(go.Bar(
        x=df_shap_imp["Mean |SHAP|"],
        y=df_shap_imp["Feature"],
        orientation="h",
        marker_color="#7F77DD",
        text=[f"{v:.4f}" for v in df_shap_imp["Mean |SHAP|"]],
        textposition="outside",
    ))
    fig_shap.update_layout(
        title="SHAP — Tầm quan trọng trung bình",
        xaxis_title="Mean |SHAP value|",
        height=380, margin=dict(l=160, r=80, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_shap, use_container_width=True)

with col_fi:
    # Feature importance của model (Gini)
    importances = model.named_steps['clf'].feature_importances_
    df_fi = pd.DataFrame({
        "Feature":    FEATURE_NAMES,
        "Importance": importances,
    }).sort_values("Importance", ascending=True)

    fig_fi = go.Figure(go.Bar(
        x=df_fi["Importance"],
        y=df_fi["Feature"],
        orientation="h",
        marker_color="#1D9E75",
        text=[f"{v:.3f}" for v in df_fi["Importance"]],
        textposition="outside",
    ))
    fig_fi.update_layout(
        title="Feature Importance (Gini — mô hình)",
        xaxis_title="Importance score",
        height=380, margin=dict(l=160, r=80, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_fi, use_container_width=True)

# ─── SHAP Scatter: phân phối SHAP từng feature ────────────────────────────────
st.markdown('<div class="section-header">SHAP theo từng feature — Hướng tác động</div>',
            unsafe_allow_html=True)

st.markdown("""
<div class="insight-box">
    Mỗi điểm = 1 khách hàng. Trục X = SHAP value (dương → tăng xác suất churn, âm → giảm).
    Màu sắc = giá trị thực của feature (đỏ = cao, xanh = thấp).
</div>
""", unsafe_allow_html=True)

# Lấy top 6 feature quan trọng nhất theo SHAP
top_features = df_shap_imp.sort_values("Mean |SHAP|", ascending=False)["Feature"].head(6).tolist()

fig_scatter = make_subplots(
    rows=2, cols=3,
    subplot_titles=top_features,
    horizontal_spacing=0.12,
    vertical_spacing=0.18,
)

for i, feat in enumerate(top_features):
    row, col = i // 3 + 1, i % 3 + 1
    feat_idx  = FEATURE_NAMES.index(feat)
    feat_vals = df_shap_sample[feat].values
    shap_col  = sv_all[:, feat_idx]

    # Normalize màu
    norm_vals = (feat_vals - feat_vals.min()) / (np.ptp(feat_vals) + 1e-9)

    fig_scatter.add_trace(go.Scatter(
        x=shap_col,
        y=np.random.uniform(-0.3, 0.3, len(shap_col)),  # jitter
        mode="markers",
        marker=dict(
            size=4,
            color=norm_vals,
            colorscale=[[0, "#378ADD"], [0.5, "#BA7517"], [1, "#E24B4A"]],
            showscale=(i == 2),
            colorbar=dict(title="Feature value<br>(chuẩn hóa)", thickness=12, len=0.4,
                          y=0.8) if i == 2 else None,
            opacity=0.6,
        ),
        hovertemplate=f"<b>{feat}</b><br>SHAP: %{{x:.4f}}<br>Value: %{{text}}<extra></extra>",
        text=[f"{v:.2f}" for v in feat_vals],
        name=feat,
    ), row=row, col=col)

    fig_scatter.add_vline(x=0, line_width=1, line_color="#aaa", row=row, col=col)

fig_scatter.update_layout(
    height=480,
    showlegend=False,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(t=60, b=40, l=40, r=60),
)
for ax in fig_scatter.layout:
    if ax.startswith("yaxis"):
        fig_scatter.layout[ax].showticklabels = False
        fig_scatter.layout[ax].showgrid = False
    if ax.startswith("xaxis"):
        fig_scatter.layout[ax].showgrid = True
        fig_scatter.layout[ax].zeroline = False

st.plotly_chart(fig_scatter, use_container_width=True)

# ─── Phân tích theo Segment ───────────────────────────────────────────────────
st.markdown('<div class="section-header">Phân tích theo phân khúc khách hàng</div>',
            unsafe_allow_html=True)

# Tạo các biến categorical để phân tích
df_seg = df_full.copy()
df_seg["Subscription Type Label"] = df_seg["Subscription Type"].map(
    {0: "Basic", 1: "Standard", 2: "Premium"})
df_seg["Contract Length Label"] = df_seg["Contract Length"].map(
    {0: "Monthly", 1: "Quarterly", 2: "Annual"})
df_seg["Gender Label"] = df_seg["Gender"].map({0: "Male", 1: "Female"})
df_seg["Age Group"] = pd.cut(df_seg["Age"],
                              bins=[17, 30, 45, 65],
                              labels=["18-30", "31-45", "46-65"])
df_seg["Tenure Group"] = pd.cut(df_seg["Tenure"],
                                 bins=[0, 12, 36, 60],
                                 labels=["< 1 năm", "1-3 năm", "> 3 năm"])

tab1, tab2, tab3, tab4 = st.tabs([
    "Subscription & Contract",
    "Nhóm tuổi & Tenure",
    "Support Calls & Payment",
    "Boxplot so sánh"
])

with tab1:
    col_sub, col_con = st.columns(2)

    with col_sub:
        df_sub_agg = (df_seg.groupby("Subscription Type Label")
                      .agg(Churn_Rate=("churn_prediction","mean"),
                           Avg_Prob=("churn_probability","mean"),
                           Count=("churn_prediction","count"))
                      .reset_index())
        fig_sub = go.Figure()
        fig_sub.add_trace(go.Bar(
            x=df_sub_agg["Subscription Type Label"],
            y=df_sub_agg["Churn_Rate"] * 100,
            marker_color=["#E24B4A","#BA7517","#1D9E75"],
            text=[f"{v*100:.1f}%" for v in df_sub_agg["Churn_Rate"]],
            textposition="outside",
            name="Churn Rate",
        ))
        fig_sub.update_layout(
            title="Churn rate theo Subscription Type",
            yaxis_title="Churn rate (%)", yaxis_range=[0, 100],
            height=320, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=50,b=30),
            showlegend=False,
        )
        st.plotly_chart(fig_sub, use_container_width=True)

    with col_con:
        df_con_agg = (df_seg.groupby("Contract Length Label")
                      .agg(Churn_Rate=("churn_prediction","mean"),
                           Count=("churn_prediction","count"))
                      .reset_index())
        fig_con = go.Figure()
        fig_con.add_trace(go.Bar(
            x=df_con_agg["Contract Length Label"],
            y=df_con_agg["Churn_Rate"] * 100,
            marker_color=["#E24B4A","#BA7517","#1D9E75"],
            text=[f"{v*100:.1f}%" for v in df_con_agg["Churn_Rate"]],
            textposition="outside",
            name="Churn Rate",
        ))
        fig_con.update_layout(
            title="Churn rate theo Contract Length",
            yaxis_title="Churn rate (%)", yaxis_range=[0, 100],
            height=320, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=50,b=30),
            showlegend=False,
        )
        st.plotly_chart(fig_con, use_container_width=True)

with tab2:
    col_age, col_tenure = st.columns(2)

    with col_age:
        df_age_agg = (df_seg.groupby("Age Group", observed=True)
                      .agg(Churn_Rate=("churn_prediction","mean"),
                           Avg_Prob=("churn_probability","mean"),
                           Count=("churn_prediction","count"))
                      .reset_index())
        fig_age = px.bar(
            df_age_agg, x="Age Group", y="Churn_Rate",
            color="Churn_Rate",
            color_continuous_scale=[[0,"#1D9E75"],[0.5,"#BA7517"],[1,"#E24B4A"]],
            text=[f"{v*100:.1f}%" for v in df_age_agg["Churn_Rate"]],
            title="Churn rate theo nhóm tuổi",
        )
        fig_age.update_traces(textposition="outside")
        fig_age.update_layout(
            yaxis_title="Churn rate", yaxis_tickformat=".0%",
            height=320, coloraxis_showscale=False,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=50,b=30),
        )
        st.plotly_chart(fig_age, use_container_width=True)

    with col_tenure:
        df_ten_agg = (df_seg.groupby("Tenure Group", observed=True)
                      .agg(Churn_Rate=("churn_prediction","mean"),
                           Count=("churn_prediction","count"))
                      .reset_index())
        fig_ten = px.bar(
            df_ten_agg, x="Tenure Group", y="Churn_Rate",
            color="Churn_Rate",
            color_continuous_scale=[[0,"#1D9E75"],[0.5,"#BA7517"],[1,"#E24B4A"]],
            text=[f"{v*100:.1f}%" for v in df_ten_agg["Churn_Rate"]],
            title="Churn rate theo thời gian sử dụng (Tenure)",
        )
        fig_ten.update_traces(textposition="outside")
        fig_ten.update_layout(
            yaxis_title="Churn rate", yaxis_tickformat=".0%",
            height=320, coloraxis_showscale=False,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=50,b=30),
        )
        st.plotly_chart(fig_ten, use_container_width=True)

with tab3:
    col_sp, col_pd = st.columns(2)

    with col_sp:
        # Scatter: Support Calls vs churn_probability
        fig_sc = px.scatter(
            df_seg.sample(min(300, len(df_seg)), random_state=42),
            x="Support Calls", y="churn_probability",
            color="risk_level",
            color_discrete_map={"Cao":"#E24B4A","Trung bình":"#BA7517","Thấp":"#1D9E75"},
            opacity=0.6, size_max=8,
            trendline="lowess",
            title="Support Calls vs Xác suất churn",
            labels={"churn_probability":"Xác suất churn","risk_level":"Rủi ro"},
        )
        fig_sc.add_hline(y=0.5, line_dash="dash", line_color="#555")
        fig_sc.update_layout(
            height=330, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=50,b=30),
            legend=dict(orientation="h",y=-0.25),
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    with col_pd:
        fig_pd = px.scatter(
            df_seg.sample(min(300, len(df_seg)), random_state=7),
            x="Payment Delay", y="churn_probability",
            color="risk_level",
            color_discrete_map={"Cao":"#E24B4A","Trung bình":"#BA7517","Thấp":"#1D9E75"},
            opacity=0.6,
            trendline="lowess",
            title="Payment Delay vs Xác suất churn",
            labels={"churn_probability":"Xác suất churn","risk_level":"Rủi ro"},
        )
        fig_pd.add_hline(y=0.5, line_dash="dash", line_color="#555")
        fig_pd.update_layout(
            height=330, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=50,b=30),
            legend=dict(orientation="h",y=-0.25),
        )
        st.plotly_chart(fig_pd, use_container_width=True)

with tab4:
    # Boxplot: churn_probability theo các nhóm categorical
    selected_seg = st.selectbox(
        "Chọn phân khúc để so sánh:",
        ["Subscription Type Label", "Contract Length Label",
         "Gender Label", "Age Group", "Tenure Group"],
        format_func=lambda x: x.replace(" Label","").replace("Label",""),
    )

    fig_box = px.box(
        df_seg, x=selected_seg, y="churn_probability",
        color=selected_seg,
        color_discrete_sequence=["#E24B4A","#BA7517","#1D9E75","#7F77DD","#378ADD"],
        points="outliers",
        title=f"Phân phối xác suất churn theo {selected_seg.replace(' Label','')}",
        labels={"churn_probability":"Xác suất churn"},
    )
    fig_box.add_hline(y=0.5, line_dash="dash", line_color="#555",
                      annotation_text="Ngưỡng 50%")
    fig_box.update_layout(
        height=400, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=60,b=40),
        showlegend=False,
    )
    st.plotly_chart(fig_box, use_container_width=True)

# ─── Insight tổng hợp ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Insight tự động</div>', unsafe_allow_html=True)

top1_feat = df_shap_imp.sort_values("Mean |SHAP|", ascending=False).iloc[0]["Feature"]
top2_feat = df_shap_imp.sort_values("Mean |SHAP|", ascending=False).iloc[1]["Feature"]

high_support = df_full[df_full["Support Calls"] >= 7]["churn_probability"].mean()
low_support  = df_full[df_full["Support Calls"] <= 2]["churn_probability"].mean()
high_delay   = df_full[df_full["Payment Delay"] >= 20]["churn_probability"].mean()

# ─── SHAP Summary (Mean |SHAP|) ───────────────────────────────────────────────
st.markdown(
    '<div class="section-header">Tầm quan trọng Features — SHAP Summary</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="insight-box">
    <b>SHAP Mean |SHAP|:</b> Trung bình giá trị tuyệt đối của SHAP trên toàn bộ dataset.
    Feature nào có giá trị cao hơn → ảnh hưởng đến dự đoán nhiều hơn.
    SHAP phản ánh mức độ tác động thực tế của từng feature lên dự đoán churn.
</div>
""", unsafe_allow_html=True)

MAX_SHAP_SAMPLES = 300

sample_idx = np.random.choice(
    len(df_enc),
    min(MAX_SHAP_SAMPLES, len(df_enc)),
    replace=False
)

df_shap_sample = df_enc.iloc[sample_idx].reset_index(drop=True)

with st.spinner(f"Đang tính SHAP trên {len(df_shap_sample)} mẫu..."):

    from sklearn.pipeline import Pipeline as _Pipeline

    # =========================================================
    # Lấy phần preprocessing trước classifier
    # =========================================================

    _preprocessor = _Pipeline(model.steps[:-1])

    # Transform dữ liệu
    df_shap_transformed = _preprocessor.transform(df_shap_sample)

    # =========================================================
    # Tính SHAP values
    # =========================================================

    shap_vals = explainer.shap_values(df_shap_transformed)

    # Xử lý nhiều format SHAP khác nhau
    if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        sv_all = shap_vals[:, :, 1]

    elif isinstance(shap_vals, list):
        sv_all = np.array(shap_vals[1])

    else:
        sv_all = shap_vals

    # =========================================================
    # Mean Absolute SHAP
    # =========================================================

    mean_abs_shap = np.abs(sv_all).mean(axis=0)

    # =========================================================
    # Feature names sau preprocessing
    # =========================================================

    try:
        transformed_feature_names = (
            _preprocessor.get_feature_names_out()
        )

    except Exception:

        transformed_feature_names = [
            f"Feature_{i}"
            for i in range(len(mean_abs_shap))
        ]

    # =========================================================
    # Đảm bảo số lượng khớp nhau
    # =========================================================

    min_len = min(
        len(transformed_feature_names),
        len(mean_abs_shap)
    )

    transformed_feature_names = (
        transformed_feature_names[:min_len]
    )

    mean_abs_shap = mean_abs_shap[:min_len]

    # =========================================================
    # DataFrame SHAP importance
    # =========================================================

    df_shap_imp = pd.DataFrame({
        "Feature": transformed_feature_names,
        "Mean |SHAP|": mean_abs_shap
    }).sort_values(
        "Mean |SHAP|",
        ascending=True
    )

# ─── DEBUG INFO ─────────────────────────────────────────────

with st.expander("🔍 Debug SHAP"):

    st.write(
        "Số feature sau preprocessing:",
        len(transformed_feature_names)
    )

    st.write(
        "Số SHAP values:",
        len(mean_abs_shap)
    )

    st.dataframe(df_shap_imp.tail(10))

# ─── SHAP BAR + FEATURE IMPORTANCE ──────────────────────────

col_shap, col_fi = st.columns(2)

# ============================================================
# SHAP Importance
# ============================================================

with col_shap:

    fig_shap = go.Figure(go.Bar(
        x=df_shap_imp["Mean |SHAP|"],
        y=df_shap_imp["Feature"],
        orientation="h",
        marker_color="#7F77DD",
        text=[
            f"{v:.4f}"
            for v in df_shap_imp["Mean |SHAP|"]
        ],
        textposition="outside",
    ))

    fig_shap.update_layout(
        title="SHAP — Tầm quan trọng trung bình",
        xaxis_title="Mean |SHAP value|",
        height=420,
        margin=dict(
            l=180,
            r=60,
            t=50,
            b=40
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(
        fig_shap,
        use_container_width=True
    )

# ============================================================
# Feature Importance của model
# ============================================================

with col_fi:

    clf = model.named_steps['clf']

    if hasattr(clf, "feature_importances_"):

        importances = clf.feature_importances_

        min_len_fi = min(
            len(importances),
            len(transformed_feature_names)
        )

        df_fi = pd.DataFrame({
            "Feature": transformed_feature_names[:min_len_fi],
            "Importance": importances[:min_len_fi],
        }).sort_values(
            "Importance",
            ascending=True
        )

        fig_fi = go.Figure(go.Bar(
            x=df_fi["Importance"],
            y=df_fi["Feature"],
            orientation="h",
            marker_color="#1D9E75",
            text=[
                f"{v:.4f}"
                for v in df_fi["Importance"]
            ],
            textposition="outside",
        ))

        fig_fi.update_layout(
            title="Feature Importance (Model)",
            xaxis_title="Importance score",
            height=420,
            margin=dict(
                l=180,
                r=60,
                t=50,
                b=40
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(
            fig_fi,
            use_container_width=True
        )

    else:

        st.info(
            "Model hiện tại không hỗ trợ feature_importances_."
        )

# ─── Insight tự động ─────────────────────────────────────────

st.markdown(
    '<div class="section-header">Insight từ SHAP</div>',
    unsafe_allow_html=True
)

top_feats = (
    df_shap_imp
    .sort_values("Mean |SHAP|", ascending=False)
    .head(5)
)

st.markdown("""
<div class="warn-box">
""", unsafe_allow_html=True)

for i, row in top_feats.iterrows():

    st.markdown(
        f"""
- <b>{row['Feature']}</b>
  ảnh hưởng mạnh đến dự đoán churn
  (Mean |SHAP| = {row['Mean |SHAP|']:.4f})
"""
        ,
        unsafe_allow_html=True
    )

st.markdown("</div>", unsafe_allow_html=True)