import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_data(csv_path):
    df = pd.read_csv(csv_path, sep="\t")
    print(f"Jumlah baris (versi Python): {len(df)}")
    return df


def clean_data(df):

    df = df.drop(df.columns[0], axis=1)

    # Rename column
    df.columns = [
        "date",
        "close",
        "volume",
        "open",
        "high",
        "low"
    ]

    # Convert date to datetime
    df["date"] = pd.to_datetime(df["date"])

    # Sort ascending (important for time series)
    df = df.sort_values("date").reset_index(drop=True)

    # Drop duplicate date if there exist
    df = df.drop_duplicates(subset="date")

    # Drop missing values
    df = df.dropna()

    print(f"Total row after cleaning: {len(df)}")
    print("Previous Date:")
    print(df[["date", "close"]].head(3))

    print("Latest Date:")
    print(df[["date", "close"]].tail(3))
    return df


def select_features(df):

    features = df[["open", "high", "low", "close"]].values
    target = df["close"].values.reshape(-1, 1)

    return features, target


def scale_data(features, target):
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    features_scaled = feature_scaler.fit_transform(features)
    target_scaled = target_scaler.fit_transform(target)

    return features_scaled, target_scaled, feature_scaler, target_scaler


def create_windowed_data(features, target, window_size=30):

    X, y = [], []

    for i in range(window_size, len(features)):
        X.append(features[i - window_size:i])
        y.append(target[i])

    X = np.array(X)
    y = np.array(y)

    print(f"Shape X: {X.shape}")
    print(f"Shape y: {y.shape}")

    return X, y


def train_test_split_time_series(X, y, train_ratio=0.8):

    split_index = int(len(X) * train_ratio)

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")

    return X_train, X_test, y_train, y_test

def create_sequences(features, target, window_size):
    X = []
    y = []

    for i in range(len(features) - window_size):
        X.append(features[i : i + window_size])
        y.append(target[i + window_size])

    return np.array(X), np.array(y)

def preprocess_pipeline(
    csv_path,
    window_size=30,
    feature_scaler=None,
    target_scaler=None,
    train=True
):
    # LOAD & CLEAN DATA

    df = load_data(csv_path)
    df = clean_data(df)

    # FEATURE & TARGET

    features = df[["open", "high", "low", "close"]].values
    target = df[["close"]].values

    # SCALING

    if train:
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()

        features = feature_scaler.fit_transform(features)
        target = target_scaler.fit_transform(target)
    else:
        features = feature_scaler.transform(features)
        target = target_scaler.transform(target)

    # WINDOWING

    X, y = create_sequences(features, target, window_size)

    # TRAIN TEST SPLIT

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return X_train, X_test, y_train, y_test, feature_scaler, target_scaler