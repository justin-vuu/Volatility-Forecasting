from data_preprocessing import load_data, compute_volatility
from model import build_lstm_model, build_bidirectional_lstm
from utils import RMSPE, RMSE
from visualize import plot_train_val_metrics, plot_model_predictions
import numpy as np

# Load and preprocess data
df = load_data('BTC-USD')
df = compute_volatility(df)

# Prepare training data
train_idx = df.index[:int(len(df)*0.7)]
val_idx = df.index[int(len(df)*0.7):int(len(df)*0.9)]
test_idx = df.index[int(len(df)*0.9):]

x_train, y_train = df['vol_current'].iloc[train_idx], df['vol_future'].iloc[train_idx]
x_val, y_val = df['vol_current'].iloc[val_idx], df['vol_future'].iloc[val_idx]
x_test, y_test = df['vol_current'].iloc[test_idx], df['vol_future'].iloc[test_idx]

# Build and train model
model = build_lstm_model(input_shape=(30, 1))  # Example: 30-day lookback window
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=200, batch_size=64)

# Visualize training and validation metrics
plot_train_val_metrics(history)

# Predictions and evaluation
y_pred = model.predict(x_val)
plot_model_predictions(y_val, y_pred, title="LSTM Model Prediction")
print(f"RMSPE: {RMSPE(y_val, y_pred)}")
print(f"RMSE: {RMSE(y_val, y_pred)}")
