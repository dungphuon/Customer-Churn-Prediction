import pandas as pd
import numpy as np

# =========================================================
# LOAD DATA
# =========================================================

print("=" * 60)
print("LOAD DATA")
print("=" * 60)

df = pd.read_csv("data_labeled.csv")

print(f"Số dòng: {len(df):,}")
print(f"Số cột : {len(df.columns)}")
print()

# =========================================================
# REQUIRED COLUMNS
# =========================================================

required_columns = [
    'Support Calls',
    'Payment Delay',
    'Last Interaction',
    'Tenure',
    'Usage Frequency',
    'Total Spend',
    'Contract Length',
    'Subscription Type'
]

missing_cols = [
    col for col in required_columns
    if col not in df.columns
]

if missing_cols:
    raise ValueError(
        f"Thiếu cột: {missing_cols}"
    )

# =========================================================
# ENCODE CATEGORICAL FEATURES
# =========================================================

print("=" * 60)
print("ENCODE CATEGORICAL FEATURES")
print("=" * 60)

contract_map = {
    'Monthly': 0,
    'Quarterly': 1,
    'Annual': 2
}

subscription_map = {
    'Basic': 0,
    'Standard': 1,
    'Premium': 2
}

# ---------------------------------------------------------
# CLEAN TEXT
# ---------------------------------------------------------

df['Contract Length'] = (
    df['Contract Length']
    .astype(str)
    .str.strip()
)

df['Subscription Type'] = (
    df['Subscription Type']
    .astype(str)
    .str.strip()
)

# ---------------------------------------------------------
# ENCODE
# ---------------------------------------------------------

df['Contract Length'] = (
    df['Contract Length']
    .replace(contract_map)
)

df['Subscription Type'] = (
    df['Subscription Type']
    .replace(subscription_map)
)

# ---------------------------------------------------------
# CONVERT TO NUMERIC
# ---------------------------------------------------------

df['Contract Length'] = pd.to_numeric(
    df['Contract Length']
)

df['Subscription Type'] = pd.to_numeric(
    df['Subscription Type']
)

print("Contract Length unique:")
print(df['Contract Length'].unique())
print()

print("Subscription Type unique:")
print(df['Subscription Type'].unique())
print()

print("Encode hoàn tất.")
print()

# =========================================================
# HANDLE MISSING VALUES
# =========================================================

print("=" * 60)
print("HANDLE MISSING VALUES")
print("=" * 60)

numeric_cols = [
    'Support Calls',
    'Payment Delay',
    'Last Interaction',
    'Tenure',
    'Usage Frequency',
    'Total Spend',
    'Contract Length',
    'Subscription Type'
]

for col in numeric_cols:

    missing_count = df[col].isnull().sum()

    if missing_count > 0:

        print(f"{col}: thiếu {missing_count}")

        median_val = df[col].median()

        df[col] = df[col].fillna(median_val)

print("Xử lý missing values hoàn tất.")
print()

# =========================================================
# GENERATE PROBABILISTIC LABELS
# =========================================================

print("=" * 60)
print("GENERATE PROBABILISTIC LABELS")
print("=" * 60)

def compute_churn_prob(row):

    # =====================================================
    # BASELINE
    # =====================================================

    score = -0.9

    # =====================================================
    # POSITIVE CHURN SIGNALS
    # =====================================================

    # Gọi support nhiều
    score += row['Support Calls'] * 0.12

    # Thanh toán trễ
    score += row['Payment Delay'] * 0.08

    # Lâu không tương tác
    score += row['Last Interaction'] * 0.02

    # =====================================================
    # NEGATIVE CHURN SIGNALS
    # =====================================================

    # Dùng lâu -> ít churn
    score -= row['Tenure'] * 0.03

    # Dùng thường xuyên -> ít churn
    score -= row['Usage Frequency'] * 0.04

    # Chi tiêu cao -> ít churn
    score -= (
        row['Total Spend'] / 1000
    ) * 0.15

    # =====================================================
    # CONTRACT EFFECT
    # =====================================================

    contract_effect = {
        0: 0.15,    # Monthly
        1: 0.05,    # Quarterly
        2: -0.10    # Annual
    }

    score += contract_effect.get(
        int(row['Contract Length']),
        0
    )

    # =====================================================
    # SUBSCRIPTION EFFECT
    # =====================================================

    subscription_effect = {
        0: 0.08,     # Basic
        1: 0.00,     # Standard
        2: -0.05     # Premium
    }

    score += subscription_effect.get(
        int(row['Subscription Type']),
        0
    )

    # =====================================================
    # INTERACTION EFFECTS
    # =====================================================

    # Support calls cao + payment delay cao
    if (
        row['Support Calls'] >= 7
        and
        row['Payment Delay'] >= 20
    ):
        score += 0.45

    # Dùng lâu + dùng thường xuyên
    if (
        row['Tenure'] >= 36
        and
        row['Usage Frequency'] >= 20
    ):
        score -= 0.35

    # Basic + Monthly
    if (
        row['Subscription Type'] == 0
        and
        row['Contract Length'] == 0
    ):
        score += 0.25

    # =====================================================
    # RANDOM BEHAVIORAL NOISE
    # =====================================================

    noise = np.random.normal(0, 0.22)

    score += noise

    # =====================================================
    # SIGMOID
    # =====================================================

    prob = 1 / (1 + np.exp(-score))

    # Tránh extreme
    prob = np.clip(prob, 0.01, 0.99)

    return prob

# =========================================================
# GENERATE LABELS
# =========================================================

np.random.seed(42)

df['churn_prob'] = df.apply(
    compute_churn_prob,
    axis=1
)

# Tung đồng xu theo probability
df['Churn'] = (
    np.random.rand(len(df))
    < df['churn_prob']
).astype(int)

# =========================================================
# DATASET STATISTICS
# =========================================================

print("=" * 60)
print("DATASET STATISTICS")
print("=" * 60)

churn_rate = df['Churn'].mean() * 100

print(f"Churn Rate: {churn_rate:.2f}%")
print()

print("Phân phối nhãn:")
print(df['Churn'].value_counts())
print()

print("Probability statistics:")
print(df['churn_prob'].describe())
print()

# =========================================================
# QUICK ANALYSIS
# =========================================================

print("=" * 60)
print("QUICK ANALYSIS")
print("=" * 60)

high_support = df[
    df['Support Calls'] >= 7
]['churn_prob'].mean()

low_support = df[
    df['Support Calls'] <= 2
]['churn_prob'].mean()

high_delay = df[
    df['Payment Delay'] >= 20
]['churn_prob'].mean()

loyal_users = df[
    (
        df['Tenure'] >= 36
    )
    &
    (
        df['Usage Frequency'] >= 20
    )
]['churn_prob'].mean()

print(
    f"Support Calls cao  -> churn prob = "
    f"{high_support:.3f}"
)

print(
    f"Support Calls thấp -> churn prob = "
    f"{low_support:.3f}"
)

print(
    f"Payment Delay cao  -> churn prob = "
    f"{high_delay:.3f}"
)

print(
    f"Loyal users        -> churn prob = "
    f"{loyal_users:.3f}"
)

print()

# =========================================================
# DROP TEMP COLUMN
# =========================================================

df_final = df.drop(
    columns=['churn_prob']
)

# =========================================================
# SAVE DATASET
# =========================================================

output_path = "customer_churn_probabilistic.csv"

df_final.to_csv(
    output_path,
    index=False
)

print("=" * 60)
print("SAVE DATASET")
print("=" * 60)

print(f"Đã lưu dataset:")
print(output_path)
print()

print("DONE")
print("=" * 60)