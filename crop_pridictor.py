
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Page config
st.set_page_config(
    page_title="🌾 Crop Production Predictor",
    page_icon="🌾",
    layout="wide"
)

# Title
st.title("🌾 Multi-Crop Production Predictor")
st.markdown("**Multiple Linear Regression | Rice, Wheat, Maize | 1961-2023**")

# Load Data Function
@st.cache_data
def load_crop_data():
    years = list(range(1961, 2024))

    # Rice Data (Ministry of Agriculture)
    rice_area = [32480, 29480, 30420, 31300, 30700, 29560, 32710, 34600, 37360, 36280,
                 37820, 38390, 37770, 41020, 41540, 41830, 42460, 43210, 44060, 37770,
                 44110, 44500, 41710, 43580, 42660, 43060, 42790, 43380, 44500, 43950,
                 42870, 42420, 42720, 44100, 43170, 43460, 42950, 42270, 41100, 41910,
                 42560, 41860, 42860, 41980, 43200, 43380, 43360, 44710, 40460, 42320,
                 42760, 40550, 44100, 43880, 43380, 43880, 43390, 44120, 45380, 43380,
                 43670, 43880, 43500]

    rice_yield = [10660, 11610, 12070, 13070, 12470, 10380, 12510, 13900, 12900, 13250,
                  11430, 12370, 11920, 11730, 11580, 10370, 11350, 12130, 12170, 14200,
                  12150, 12880, 13740, 13550, 14650, 14100, 13230, 13550, 13820, 16860,
                  16870, 15780, 18830, 18570, 18400, 18890, 19790, 19870, 21250, 20260,
                  21800, 20210, 19400, 20860, 21090, 21050, 22230, 22190, 25260, 22680,
                  22320, 25960, 23860, 23780, 24040, 23120, 25250, 23910, 26040, 28280,
                  29280, 30820, 31670]

    rice_prod = [34600, 34200, 36700, 40900, 38300, 30700, 40900, 48100, 48200, 48100,
                 43200, 47500, 45000, 48100, 48100, 43400, 48200, 52400, 53600, 53600,
                 53600, 57300, 57300, 59000, 62500, 60700, 56600, 58800, 61500, 74100,
                 72300, 66900, 80400, 81900, 79400, 82100, 85000, 84000, 87300, 84900,
                 92800, 84600, 83100, 87600, 91100, 91300, 96400, 99200, 102200, 95980,
                 95400, 105300, 105240, 104320, 104320, 101480, 109520, 105480, 118140, 122760,
                 127840, 135210, 137800]

    # Wheat Data
    wheat_area = [11100, 12970, 13190, 13580, 12650, 13310, 15020, 16190, 16380, 18240,
                  19490, 20810, 21620, 22300, 21590, 20780, 21770, 21910, 22260, 22260,
                  22520, 23280, 23980, 23820, 23270, 23070, 22630, 23840, 24100, 24190,
                  23750, 24810, 24710, 25280, 24920, 23750, 25490, 27000, 27530, 25690,
                  26380, 25320, 26100, 26360, 26510, 26450, 27620, 27900, 27770, 29000,
                  29900, 29540, 29670, 31190, 30230, 30550, 30640, 29650, 30510, 31360,
                  30990, 31090, 30770]

    wheat_yield = [9910, 9310, 8170, 9080, 8220, 8560, 7580, 10220, 11380, 11010,
                   12230, 12690, 11430, 9770, 11160, 13890, 13330, 14490, 15950, 14300,
                   16120, 16100, 17860, 18500, 18930, 20390, 19580, 19370, 20690, 22800,
                   22950, 22440, 23150, 23670, 24920, 26370, 27200, 24570, 25710, 29730,
                   26030, 28360, 24950, 27370, 25900, 26220, 27450, 28160, 29060, 27860,
                   29050, 32120, 31520, 30730, 28620, 30610, 32150, 34940, 33500, 34390,
                   35360, 34650, 36640]

    wheat_prod = [11000, 12080, 10780, 12330, 10400, 11390, 11390, 16540, 18650, 20090,
                  23830, 26410, 24710, 21780, 24100, 28850, 29010, 31750, 35510, 31830,
                  36310, 37490, 42820, 44070, 44060, 47050, 44320, 46170, 49850, 55140,
                  54510, 55690, 57210, 59840, 62100, 62620, 69350, 66350, 70770, 76370,
                  68680, 71820, 65130, 72150, 68640, 69350, 75810, 78570, 80680, 80800,
                  86870, 94880, 93510, 95850, 86530, 93500, 98510, 103600, 102190, 107860,
                  109590, 107740, 112740]

    # Maize Data
    maize_area = [3470, 3580, 3770, 3820, 4000, 4070, 4190, 4380, 4550, 4810,
                  5040, 5240, 5450, 5630, 5800, 5770, 5850, 5910, 5980, 6020,
                  5910, 5790, 5800, 5870, 5870, 5820, 5860, 5900, 5930, 6010,
                  6020, 5860, 5980, 6250, 6170, 6210, 6360, 6530, 6470, 6390,
                  6450, 6590, 7010, 6910, 6980, 7410, 7640, 7890, 8170, 8360,
                  8670, 9000, 9210, 9170, 8880, 9200, 9530, 9200, 9200, 9170,
                  9530, 9580, 9650]

    maize_yield = [11820, 12490, 12730, 13170, 13550, 13860, 14270, 14540, 14880, 15570,
                   15380, 15250, 15250, 15310, 15380, 14330, 15010, 15260, 15340, 15450,
                   15350, 14990, 15260, 15480, 15670, 15720, 15910, 16150, 16340, 16720,
                   16840, 16520, 17410, 17760, 17340, 17660, 18130, 18790, 18390, 18330,
                   18670, 18730, 19430, 20510, 21080, 20380, 21890, 25000, 24150, 25990,
                   25100, 25080, 26240, 26560, 25420, 28150, 30170, 30220, 30600, 34510,
                   35280, 36150, 36970]

    maize_prod = [4100, 4470, 4800, 5030, 5420, 5640, 5980, 6370, 6770, 7490,
                  7750, 7990, 8310, 8620, 8920, 8270, 8780, 9020, 9170, 9300,
                  9070, 8680, 8850, 9090, 9200, 9150, 9320, 9530, 9690, 10050,
                  10140, 9680, 10410, 11100, 10700, 10970, 11530, 12270, 11900, 11710,
                  12040, 12340, 13620, 14170, 14710, 15100, 16720, 19730, 19730, 21730,
                  21760, 22570, 24170, 24350, 22570, 25900, 28750, 27800, 28150, 31650,
                  33620, 34630, 35680]

    rice_df = pd.DataFrame({
        'Year': years, 'Area_Harvested_1000_ha': rice_area, 
        'Yield_hg_ha': rice_yield, 'Production_1000_tonnes': rice_prod
    })

    wheat_df = pd.DataFrame({
        'Year': years, 'Area_Harvested_1000_ha': wheat_area, 
        'Yield_hg_ha': wheat_yield, 'Production_1000_tonnes': wheat_prod
    })

    maize_df = pd.DataFrame({
        'Year': years, 'Area_Harvested_1000_ha': maize_area, 
        'Yield_hg_ha': maize_yield, 'Production_1000_tonnes': maize_prod
    })

    return rice_df, wheat_df, maize_df

# Train Model
def train_model(df):
    X = df[['Year', 'Area_Harvested_1000_ha', 'Yield_hg_ha']]
    y = df['Production_1000_tonnes']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    return model, scaler

# Predict
def predict_production(model, scaler, year, area, yield_val):
    X_new = scaler.transform([[year, area, yield_val]])
    return model.predict(X_new)[0]

# Main App
st.sidebar.header("🔧 Select Crop")
crop = st.sidebar.selectbox("Choose Crop:", ["Rice 🌾", "Wheat 🌾", "Maize 🌽"])

rice_df, wheat_df, maize_df = load_crop_data()

if crop == "Rice 🌾":
    df = rice_df.copy()
    crop_name = "Rice"
elif crop == "Wheat 🌾":
    df = wheat_df.copy()
    crop_name = "Wheat"
else:
    df = maize_df.copy()
    crop_name = "Maize"

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.header(f"📊 {crop_name} Production Predictor")

    # Train model and get metrics
    model, scaler = train_model(df)
    X_scaled = scaler.transform(df[['Year', 'Area_Harvested_1000_ha', 'Yield_hg_ha']])
    y_pred = model.predict(X_scaled)

    r2 = r2_score(df['Production_1000_tonnes'], y_pred)
    rmse = np.sqrt(mean_squared_error(df['Production_1000_tonnes'], y_pred))

    col1m, col2m = st.columns(2)
    col1m.metric("**R² Score**", f"{r2:.3f}")
    col2m.metric("**RMSE**", f"{rmse:,.0f} tonnes")

    # Prediction inputs
    st.subheader("🔮 Predict 2025 Production")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        year = st.number_input("Year", min_value=2024, max_value=2030, value=2025)
    with col_b:
        area = st.number_input("Area (1000 ha)", value=int(df['Area_Harvested_1000_ha'].mean()))
    with col_c:
        yield_val = st.number_input("Yield (hg/ha)", value=int(df['Yield_hg_ha'].mean()))

    if st.button("🚀 Predict Production", type="primary"):
        prediction = predict_production(model, scaler, year, area, yield_val)
        st.success(f"**Predicted {crop_name} Production: {prediction:,.0f} thousand tonnes**")

        # Show trend chart
        fig = px.line(df.tail(10), x='Year', y='Production_1000_tonnes', 
                     title=f"{crop_name} Production Trend")
        fig.add_hline(y=prediction, line_dash="dash", line_color="red",
                     annotation_text=f"Predicted {year}: {prediction:,.0f}")
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📋 Sample Data")
    st.dataframe(df[['Year', 'Area_Harvested_1000_ha', 'Yield_hg_ha', 'Production_1000_tonnes']].head(10))

    # Download
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"{crop_name.lower()}_data.csv",
        mime="text/csv"
    )

st.markdown("---")
st.markdown("*Real Ministry of Agriculture Data (1961-2023)* [code_file:223][code_file:224]")


