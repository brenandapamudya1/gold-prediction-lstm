from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def build_lstm_model(input_shape):
    """
    Load model for LSTM
    """

    model = Sequential()

    model.add(
        LSTM(
            units=32,
            input_shape=input_shape
        )
    )

    model.add(Dense(1))

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model
