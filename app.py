
import base64
from pathlib import Path

import pandas as pd
import streamlit as st
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(
    page_title="Prediksi Harga iPhone",
    page_icon="📱",
    layout="wide"
)

DATA_PATH = Path("iphoneFeaturesPriceDataset.csv")
IMAGE_PATH = Path("assets/iphone_store_estetik.jpeg")

def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg64 = img_to_base64(IMAGE_PATH)

st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{
    background:
        linear-gradient(135deg, rgba(6, 10, 22, 0.92), rgba(15, 23, 42, 0.74)),
        url("data:image/jpeg;base64,{bg64}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
}}

.block-container {{
    padding-top: 2rem;
    padding-bottom: 2rem;
}}

.hero-card {{
    border-radius: 28px;
    padding: 38px;
    background: linear-gradient(135deg, rgba(255,255,255,0.20), rgba(255,255,255,0.07));
    border: 1px solid rgba(255,255,255,0.22);
    box-shadow: 0 28px 80px rgba(0,0,0,0.42);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
}}

.hero-title {{
    font-size: 52px;
    font-weight: 900;
    color: white;
    line-height: 1.05;
    margin-bottom: 12px;
}}

.hero-subtitle {{
    font-size: 18px;
    color: rgba(255,255,255,0.86);
    max-width: 780px;
}}

.glass-box {{
    border-radius: 22px;
    padding: 24px;
    background: rgba(255,255,255,0.13);
    border: 1px solid rgba(255,255,255,0.18);
    box-shadow: 0 18px 55px rgba(0,0,0,0.35);
    backdrop-filter: blur(14px);
}}

.big-price {{
    font-size: 42px;
    font-weight: 900;
    color: #ffffff;
    margin-top: 5px;
}}

.small-label {{
    color: white;
    font-size: 14px;
}}
h1, h2, {{
    color: white;
    }}
div{{
    color: black;
    }}

[data-testid="stWidgetLabel"] {{
    color: white !important;
}}

div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stTextInput"] label {{
    color: white !important;
    font-weight: 600;
}}


h3{{
    color: black;
    }}

p{{
    color: #white;
}}
div.st.subheader{{
color: white;
}}

[data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{
    color: white !important;
}}

div.stButton > button {{
    border-radius: 14px;
    color: black;
    border: none;
    font-weight: 800;
    padding: 0.7rem 1.2rem;
}}

div.stButton > button:hover {{
    color: white;
}}

[data-testid="stSidebar"] {{
    background: white;
}}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<style>

[data-testid="stSidebar"] {{
    background: rgba(2, 6, 23, 0.88);
}}

/* TAB */
.stTabs [data-baseweb="tab"] {{
    color: white !important;
    font-weight: bold;
}}

.stTabs [aria-selected="true"] {{
    color: white !important;
}}

</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

target_col = None
for c in df.columns:
    if str(c).lower() in ["price", "harga", "price in inr", "sale price", "mrp", "current price"]:
        target_col = c
        break
if target_col is None:
    target_col = df.columns[-1]

# Convert possible target strings to numeric
work_df = df.copy()
work_df[target_col] = (
    work_df[target_col]
    .astype(str)
    .str.replace(r"[^0-9.\-]", "", regex=True)
    .replace("", np.nan)
)
work_df[target_col] = pd.to_numeric(work_df[target_col], errors="coerce")
work_df = work_df.dropna(subset=[target_col])

st.markdown("""
<div class="hero-card">
    <div class="hero-title">Prediksi Harga iPhone</div>
    <div class="hero-subtitle">
        Aplikasi machine learning untuk memprediksi harga iPhone berdasarkan fitur dataset.
        Desain dibuat lebih estetik dengan tampilan Apple Store dan efek glassmorphism.
    </div>
</div>
""", unsafe_allow_html=True)

st.write("")

colA, colB, colC = st.columns(3)
with colA:
    st.markdown('<div class="glass-box"><div class="small-label">Jumlah Data</div><div class="big-price">{}</div></div>'.format(len(work_df)), unsafe_allow_html=True)
with colB:
    st.markdown('<div class="glass-box"><div class="small-label">Jumlah Kolom</div><div class="big-price">{}</div></div>'.format(len(work_df.columns)), unsafe_allow_html=True)
with colC:
    st.markdown('<div class="glass-box"><div class="small-label">Target Prediksi</div><div class="big-price" style="font-size:30px;">{}</div></div>'.format(target_col), unsafe_allow_html=True)

st.write("")

tab1, tab2, tab3 = st.tabs(["📱 Prediksi", "📊 Dataset", "🤖 Evaluasi Model"])

feature_cols = [c for c in work_df.columns if c != target_col]
X = work_df[feature_cols]
y = work_df[target_col]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=150, random_state=42))
])

if len(work_df) >= 5:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
else:
    X_train, X_test, y_train, y_test = X, X, y, y

model.fit(X_train, y_train)

# with tab1:
#     st.subheader("Masukkan Spesifikasi iPhone")
with tab1:
    st.markdown(
        "<h3 style='color:white;'>Masukkan Spesifikasi iPhone</h3>",
        unsafe_allow_html=True
    )

    input_data = {}
    cols_input = st.columns(2)

    for i, col in enumerate(feature_cols):
        with cols_input[i % 2]:
            if col in numeric_features:
                min_val = float(pd.to_numeric(work_df[col], errors="coerce").min())
                max_val = float(pd.to_numeric(work_df[col], errors="coerce").max())
                mean_val = float(pd.to_numeric(work_df[col], errors="coerce").mean())
                if np.isnan(min_val) or np.isnan(max_val) or np.isnan(mean_val):
                    input_data[col] = st.number_input(col, value=0.0)
                else:
                    input_data[col] = st.number_input(col, min_value=min_val, max_value=max_val, value=mean_val)
            else:
                options = sorted(work_df[col].astype(str).dropna().unique().tolist())
                if len(options) == 0:
                    input_data[col] = st.text_input(col, value="")
                else:
                    input_data[col] = st.selectbox(col, options)

    if st.button("Prediksi Harga Sekarang"):
        input_df = pd.DataFrame([input_data])
        pred = model.predict(input_df)[0]
        st.markdown(f"""
        <div class="glass-box">
            <div class="small-label">Estimasi Harga iPhone</div>
            <div class="big-price">Rp {pred:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.subheader("Preview Dataset")
    st.dataframe(work_df, use_container_width=True)

with tab3:
    st.subheader("Evaluasi Model")
    pred_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred_test)
    r2 = r2_score(y_test, pred_test) if len(y_test) > 1 else 0

    c1, c2 = st.columns(2)
    c1.metric("MAE", f"{mae:,.2f}")
    c2.metric("R² Score", f"{r2:.3f}")

    eval_df = pd.DataFrame({
        "Harga Asli": y_test.values,
        "Harga Prediksi": pred_test
    })
    st.dataframe(eval_df, use_container_width=True)

st.caption("Dibuat dengan Streamlit, scikit-learn, dan desain estetik berbasis gambar iPhone.")
