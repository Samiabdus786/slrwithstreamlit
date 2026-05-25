import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Laptop Price Prediction",
    page_icon="💻",
    layout="wide"
)

st.markdown("""
<style>

.main {
    background-color: #0f172a;
}

.main-title {
    background: linear-gradient(135deg,#2563eb,#7c3aed);
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    color: white;
    margin-bottom: 25px;
}

.card {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 18px;
    margin-bottom: 20px;
    color: white;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.3);
}

.metric-card {
    background: linear-gradient(135deg,#16a34a,#22c55e);
    padding: 18px;
    border-radius: 15px;
    text-align: center;
    color: white;
    font-size: 22px;
    font-weight: bold;
}

.prediction-box {
    background: linear-gradient(135deg,#f59e0b,#ea580c);
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    color: white;
    font-size: 30px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-title">
<h1>💻 Laptop Price Prediction</h1>
<p>Machine Learning Project using Linear Regression</p>
</div>
""", unsafe_allow_html=True)

np.random.seed(42)

brands = ["HP", "Dell", "Lenovo", "Asus", "Acer", "Apple"]
processors = ["i3", "i5", "i7", "Ryzen 5", "Ryzen 7"]
rams = [4, 8, 16, 32]
ssds = [128, 256, 512, 1024]

data = []

for i in range(600):

    brand = np.random.choice(brands)
    processor = np.random.choice(processors)
    ram = np.random.choice(rams)
    ssd = np.random.choice(ssds)

    price = 25000

    if processor == "i5":
        price += 12000

    elif processor == "i7":
        price += 25000

    elif processor == "Ryzen 5":
        price += 15000

    elif processor == "Ryzen 7":
        price += 28000

    price += ram * 3000
    price += ssd * 45

    if brand == "Apple":
        price += 50000

    elif brand == "Dell":
        price += 8000

    elif brand == "HP":
        price += 5000

    price += np.random.randint(-5000, 5000)

    data.append([brand, processor, ram, ssd, price])

df = pd.DataFrame(
    data,
    columns=["Brand", "Processor", "RAM", "SSD", "Price"]
)

st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("📋 Dataset Preview")

st.dataframe(df.head(10), use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

label_encoders = {}

for col in ["Brand", "Processor"]:

    le = LabelEncoder()

    df[col] = le.fit_transform(df[col])

    label_encoders[col] = le

X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)

adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)

st.markdown("## 📊 Model Performance")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card">
    MAE<br>{mae:,.0f}
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
    RMSE<br>{rmse:,.0f}
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
    R² Score<br>{r2:.3f}
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
    Adj R²<br>{adj_r2:.3f}
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("🔥 Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(7,5))

    sns.heatmap(
        df.corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)

with col2:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📈 Actual vs Predicted")

    fig2 = px.scatter(
        x=y_test,
        y=y_pred,
        labels={
            "x": "Actual Price",
            "y": "Predicted Price"
        },
        title="Actual vs Predicted Prices"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

st.markdown("""
<div class="card">
<h2>💻 Predict Laptop Price</h2>
</div>
""", unsafe_allow_html=True)

col3, col4 = st.columns(2)

with col3:

    brand = st.selectbox(
        "Select Brand",
        brands
    )

    processor = st.selectbox(
        "Select Processor",
        processors
    )

with col4:

    ram = st.selectbox(
        "Select RAM (GB)",
        rams
    )

    ssd = st.selectbox(
        "Select SSD (GB)",
        ssds
    )

brand_encoded = label_encoders["Brand"].transform([brand])[0]

processor_encoded = label_encoders["Processor"].transform([processor])[0]

input_data = pd.DataFrame(
    [[brand_encoded, processor_encoded, ram, ssd]],
    columns=["Brand", "Processor", "RAM", "SSD"]
)

input_scaled = scaler.transform(input_data)

prediction = model.predict(input_scaled)[0]

st.markdown(f"""
<div class="prediction-box">
💰 Predicted Laptop Price <br><br>
₹ {prediction:,.0f}
</div>
""", unsafe_allow_html=True)

st.markdown("---")

st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("📌 Model Equation")

coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

st.dataframe(coef_df, use_container_width=True)

st.write(f"### Intercept: {model.intercept_:,.2f}")

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center;
padding:15px;
font-size:18px;
color:gray;'>
❤️ Built with Streamlit + Linear Regression
</div>
""", unsafe_allow_html=True)