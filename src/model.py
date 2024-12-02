import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, InputLayer
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def build_lstm_model(input_shape, n_units=20):
    model = Sequential([
        InputLayer(input_shape=input_shape),
        LSTM(n_units),
        Dense(1)
    ])
    
    model.compile(loss='mse', optimizer="adam", metrics=['mse'])
    return model

def build_bidirectional_lstm(input_shape, n_units_1=32, n_units_2=16):
    model = Sequential([
        InputLayer(input_shape=input_shape),
        Bidirectional(LSTM(n_units_1, return_sequences=True)),
        Bidirectional(LSTM(n_units_2)),
        Dense(1)
    ])
    
    model.compile(loss='mse', optimizer="adam", metrics=['mse'])
    return model
