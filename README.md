# Gold Price Prediction using LSTM

This project implements a **Long Short-Term Memory (LSTM) neural network** to predict **gold prices** based on historical time series data using **Python and TensorFlow**.

The model is trained on past gold price data and predicts the **next-day closing price**.

---

## 📌 Features

- Time series data preprocessing (cleaning, scaling, windowing)
- LSTM-based regression model
- Separate training and prediction scripts
- Exploratory Data Analysis (EDA) using Jupyter Notebook
- Model and scalers saved for inference

---

## 📂 Project Structure

```bash
gold-prices-nn/
│
├── data/
│   └── gold_prices.csv          # Gold price dataset
│
├── src/
│   ├── preprocessing.py         # Data preprocessing pipeline
│   ├── model.py                 # LSTM model architecture
│   ├── train.py                 # Model training script
│   └── predict.py               # Price prediction script
│
├── notebooks/
│   └── exploration.ipynb        # Exploratory Data Analysis (EDA)
│
├── models/
│   ├── gold_price_lstm.keras    # Trained LSTM model
│   ├── feature_scaler.pkl       # Feature scaler
│   └── target_scaler.pkl        # Target scaler
│
├── requirements.txt             # Project dependencies
└── README.md

## 📊 Dataset

Source: Historical gold price data
Columns:
-Date
-Open
-High
-Low
-Close
-Volume
The dataset is sorted in ascending chronological order to ensure proper time series learning.
