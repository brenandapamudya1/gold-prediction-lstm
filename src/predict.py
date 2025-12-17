import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from preprocessing import clean_data

WINDOW_SIZE = 30


def load_latest_window(csv_path, feature_scaler):
    df = pd.read_csv(csv_path, sep=",")
    df = clean_data(df)
    features = df[["open", "high", "low", "close"]].values
    features_scaled = feature_scaler.transform(features)

    last_window = features_scaled[-WINDOW_SIZE:]
    return np.expand_dims(last_window, axis=0)


def main():
    model = load_model("models/gold_price_lstm.keras")
    feature_scaler = joblib.load("models/feature_scaler.pkl")
    target_scaler = joblib.load("models/target_scaler.pkl")

    X_input = load_latest_window(
        csv_path="data/gold_prices.csv",
        feature_scaler=feature_scaler
    )

    y_pred_scaled = model.predict(X_input)
    y_pred = target_scaler.inverse_transform(y_pred_scaled)

    print("===== GOLD PRICE PREDICTION =====")
    print(f"Gold price prediction TOMORROW: {y_pred[0][0]:.2f} USD")


if __name__ == "__main__":
    main()
