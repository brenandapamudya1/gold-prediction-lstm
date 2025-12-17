import os
import joblib

from preprocessing import preprocess_pipeline
from model import build_lstm_model


def main():
    # PREPROCESSING
    X_train, X_test, y_train, y_test, f_scaler, t_scaler = preprocess_pipeline(
        csv_path="data/gold_prices.csv",
        window_size=30
    )

    # BUILD MODEL
    model = build_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2])
    )

    # TRAIN MODEL
    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )

    # SAVE MODEL & SCALER
    os.makedirs("models", exist_ok=True)

    model.save("models/gold_price_lstm.keras")
    joblib.dump(f_scaler, "models/feature_scaler.pkl")
    joblib.dump(t_scaler, "models/target_scaler.pkl")

    print("Model and scaler succesfully saved")


if __name__ == "__main__":
    main()
