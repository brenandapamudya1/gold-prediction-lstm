import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import matplotlib
matplotlib.use("Agg")

from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

from preprocessing import preprocess_pipeline

def main():
    # PATH
    csv_path = "data/gold_prices.csv"
    model_path = "models/gold_price_lstm.keras"
    feature_scaler_path = "models/feature_scaler.pkl"
    target_scaler_path = "models/target_scaler.pkl"

    # LOAD MODEL & SCALER
    model = load_model(model_path)
    feature_scaler = joblib.load(feature_scaler_path)
    target_scaler = joblib.load(target_scaler_path)

    # PREPROCESS DATA
    X_train, X_test, y_train, y_test, _, _ = preprocess_pipeline(
        csv_path=csv_path,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        train=False
    )

    # PREDICTION
    y_pred = model.predict(X_test)

    # INVERSE SCALING
    y_test_inv = target_scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_inv = target_scaler.inverse_transform(y_pred)

    # ===== EVALUATION METRICS =====
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mape = np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100

    print("===== MODEL EVALUATION =====")
    print(f"MAE  : {mae:.2f} USD")
    print(f"RMSE : {rmse:.2f} USD")
    rint(f"MAPE : {mape:.2f}%")
    
    # ===== PLOT =====
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_inv, label="Actual Price")
    plt.plot(y_pred_inv, label="Predicted Price")
    plt.title("Gold Price Prediction (Test Set)")
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/prediction_vs_actual.png")
    plt.close()


if __name__ == "__main__":
    main()
