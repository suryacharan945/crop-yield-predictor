# 🌾 Crop Yield & Weather Predictor (2000–2025)

This Streamlit app predicts crop yield, temperature, and rainfall for Indian states using historical weather data and machine learning models.

## 📁 Files

- `app.py` — Streamlit frontend
- `data/combined_crop_weather_dataset.csv` — Historical data
- `data/future_yield_predictions.csv` — Predicted yield for 2023–2025
- `isi_project_codes.pdf` — Original code and training pipeline

## 📊 Features

- Predict yield from 2000 to 2025
- Weather modeling using Random Forest
- Yield forecasting using Gradient Boosting
- Interactive visualizations and trends
- Handles edge cases like crop name mismatch and rainfall scaling

## 🚀 How to Run

```bash
pip install -r requirements.txt
python -m streamlit run app.py



