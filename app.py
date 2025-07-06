import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="centered", page_title="Crop Yield & Weather Predictor", page_icon="🌾")

@st.cache_data
def load_data():
    df_hist = pd.read_csv("combined_crop_weather_dataset.csv")
    df_pred = pd.read_csv("future_yield_predictions.csv")

    # Normalize column names
    df_hist.columns = df_hist.columns.str.strip().str.replace(" ", "_").str.lower()
    df_pred.columns = df_pred.columns.str.strip().str.replace(" ", "_").str.lower()

    # Parse and extract year for historical
    df_hist['temperature_recorded_date'] = pd.to_datetime(df_hist['temperature_recorded_date'], errors='coerce')
    df_hist['year'] = df_hist['temperature_recorded_date'].dt.year

    # Standardize crop/state names
    df_hist['crop'] = df_hist['crop'].str.strip().str.title()
    df_hist['state_name'] = df_hist['state_name'].str.strip().str.title()
    df_pred['crop'] = df_pred['crop'].str.strip().str.title()
    df_pred['state'] = df_pred['state'].str.strip().str.title()

    # Map mismatched crops from historical to prediction dataset naming
    crop_name_mapping = {
        "Rabi Rice": "rice",
        "Mustard": "rapeseed &mustard"
    }
    df_pred['crop'] = df_pred['crop'].replace({v: v for v in df_pred['crop'].unique()})
    df_hist['mapped_crop'] = df_hist['crop'].replace(crop_name_mapping)

    # Fix rainfall scale (if total_rainfall is in mm, divide by 1000)
    df_pred['total_rainfall'] = df_pred['total_rainfall'] / 1000.0

    return df_hist, df_pred

df_hist, df_pred = load_data()

# === UI ===
st.markdown("## 🌾 Crop Yield & Weather Predictor (2000–2025)")
st.markdown("Predict average annual yield, rainfall, and temperature.")

crop_options = sorted(df_hist['crop'].dropna().unique())
state_options = sorted(df_hist['state_name'].dropna().unique())
year_options = list(range(2000, 2026))

selected_crop = st.selectbox("Select Crop", crop_options)
selected_state = st.selectbox("Select State", state_options)
selected_year = st.selectbox("Select Year", year_options, index=year_options.index(2022))

# === Processing ===
output = {}
record_found = False
predicted = False

# Apply crop name mapping for prediction lookup
crop_name_mapping = {
    "Rabi Rice": "Rice",
    "Mustard": "Rapeseed &Mustard"
}
mapped_crop = crop_name_mapping.get(selected_crop, selected_crop)

if selected_year <= 2022:
    filtered = df_hist[
        (df_hist['crop'] == selected_crop) &
        (df_hist['state_name'] == selected_state) &
        (df_hist['year'] == selected_year)
    ]
    if not filtered.empty:
        record_found = True
        output = {
            "Avg Rainfall (mm)": round(filtered['state_rainfall_val'].mean(), 2),
            "Avg Max Temp (°C)": round(filtered['state_temperature_max_val'].mean(), 2),
            "Avg Min Temp (°C)": round(filtered['state_temperature_min_val'].mean(), 2),
            "Avg Yield (tonnes/ha)": round(filtered['yield'].mean(), 2)
        }

else:
    filtered = df_pred[
        (df_pred['crop'] == mapped_crop) &
        (df_pred['state'] == selected_state) &
        (df_pred['year'] == selected_year)
    ]
    if not filtered.empty:
        predicted = True
        output = {
            "Predicted Rainfall (m)": round(filtered['total_rainfall'].mean(), 3),
            "Predicted Max Temp (°C)": round(filtered['avg_max_temp'].mean(), 2),
            "Predicted Min Temp (°C)": round(filtered['avg_min_temp'].mean(), 2),
            "Predicted Yield (tonnes/ha)": round(filtered['predicted_yield'].mean(), 2)
        }

# === Display ===
st.markdown("---")
if record_found:
    st.success("✅ Historical yearly average data found.")
elif predicted:
    st.info("🔮 Predicted yearly average data displayed.")
else:
    st.warning("⚠️ No data found for the selected combination.")

for key, value in output.items():
    st.metric(key, f"{value}")

# === Visualization ===
st.markdown("---")
st.subheader("📊 Visualize Yield Trends")

with st.expander("📈 Historical Yield Trends (2000–2022)"):
    hist_crop = st.selectbox("Select crop for trend:", crop_options, key="hist_crop")
    hist_state = st.selectbox("Select state:", state_options, key="hist_state")

    hist_filtered = df_hist[
        (df_hist['crop'] == hist_crop) &
        (df_hist['state_name'] == hist_state)
    ].dropna(subset=['year', 'yield'])

    if not hist_filtered.empty:
        yearly_avg = hist_filtered.groupby('year')['yield'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=yearly_avg, x='year', y='yield', marker='o', ax=ax)
        ax.set_title(f"Historical Average Yield - {hist_crop} ({hist_state})")
        ax.set_xlabel("Year")
        ax.set_ylabel("Yield (tonnes/ha)")
        st.pyplot(fig)
    else:
        st.warning("No historical data found for selected crop and state.")

with st.expander("🔮 Future Yield Predictions (2023–2025)"):
    fut_crop = st.selectbox("Select crop for predictions:", crop_options, key="fut_crop")
    fut_state = st.selectbox("Select state:", state_options, key="fut_state")
    fut_mapped_crop = crop_name_mapping.get(fut_crop, fut_crop)

    fut_filtered = df_pred[
        (df_pred['crop'] == fut_mapped_crop) &
        (df_pred['state'] == fut_state)
    ]

    if not fut_filtered.empty:
        future_avg = fut_filtered.groupby('year')['predicted_yield'].mean().reset_index()
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=future_avg, x='year', y='predicted_yield', marker='o', ax=ax2)
        ax2.set_title(f"Predicted Yield - {fut_crop} ({fut_state})")
        ax2.set_xlabel("Year")
        ax2.set_ylabel("Predicted Yield (tonnes/ha)")
        st.pyplot(fig2)
    else:
        st.warning("No prediction data found for selected crop and state.")
