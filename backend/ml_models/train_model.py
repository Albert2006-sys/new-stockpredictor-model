import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

# Import from your project structure
from ml_models.feature_engineering import create_features
# from utils.database_manager import DatabaseManager # Uncomment when ready

# --- 1. Triple-Barrier Labeling ---
# ... (This section contains the complex logic for creating high-quality labels) ...
def get_daily_volatility(close, lookback=100):
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(close.index[df0 - 1], index=close.index[close.shape[0] - len(df0):])
    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1
    df0 = df0.ewm(span=lookback).std()
    return df0

def apply_triple_barrier(close, events, pt_sl, molecule):
    out = events[['t1']].copy(deep=True)
    if pt_sl[0] > 0: pt = pt_sl[0] * events['trgt']
    else: pt = pd.Series(index=events.index)
    if pt_sl[1] > 0: sl = -pt_sl[1] * events['trgt']
    else: sl = pd.Series(index=events.index)
    for loc, t1 in events.loc[molecule, 't1'].fillna(close.index[-1]).iteritems():
        df0 = close[loc:t1]
        df0 = (df0 / close[loc] - 1)
        out.loc[loc, 'sl'] = df0[df0 < sl[loc]].index.min()
        out.loc[loc, 'pt'] = df0[df0 > pt[loc]].index.min()
    out['t1'] = pd.to_datetime(out['t1'])
    return out

def get_labels(close_prices, time_horizon, pt_sl_multipliers, vol_lookback=100):
    events = pd.DataFrame(index=close_prices.index)
    t1 = close_prices.index.searchsorted(close_prices.index + pd.Timedelta(minutes=time_horizon))
    t1 = t1[t1 < close_prices.shape[0]]
    t1 = pd.Series(close_prices.index[t1], index=close_prices.index[:t1.shape[0]])
    events['t1'] = t1
    vol = get_daily_volatility(close_prices, lookback=vol_lookback)
    events['trgt'] = vol.reindex(events.index, method='ffill')
    events = events.dropna(subset=['trgt'])
    first_touch_times = apply_triple_barrier(close_prices, events, pt_sl_multipliers, events.index)
    labels = pd.DataFrame(index=events.index)
    labels['label'] = 0
    pt_win = first_touch_times.pt.notna() & (first_touch_times.pt <= first_touch_times.t1)
    pt_win = pt_win & (first_touch_times.sl.isna() | (first_touch_times.pt < first_touch_times.sl))
    labels.loc[pt_win, 'label'] = 2 # UP
    sl_loss = first_touch_times.sl.notna() & (first_touch_times.sl <= first_touch_times.t1)
    sl_loss = sl_loss & (first_touch_times.pt.isna() | (first_touch_times.sl < first_touch_times.pt))
    labels.loc[sl_loss, 'label'] = 0 # DOWN
    # Note: 1 is NEUTRAL
    return labels.dropna()

# --- 2. Data Preparation ---
def create_sequences(data, labels, sequence_length=60):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data.iloc[i:(i + sequence_length)].values)
        y.append(labels.iloc[i + sequence_length])
    return np.array(X), np.array(y)

# --- 3. Model Architectures ---
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x + inputs)
    ff_out = Dense(ff_dim, activation="relu")(x)
    ff_out = Dense(inputs.shape[-1])(ff_out)
    ff_out = Dropout(dropout)(ff_out)
    return LayerNormalization(epsilon=1e-6)(x + ff_out)

def build_transformer_model(input_shape, head_size, num_heads, ff_dim, num_blocks, mlp_units, dropout=0, mlp_dropout=0):
    inputs = Input(shape=input_shape)
    x = inputs
    for _ in range(num_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    x = GlobalAveragePooling1D(data_format="channels_last")(x)
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)
    outputs = Dense(3, activation="softmax")(x)
    return Model(inputs, outputs)

def build_lstm_model(input_shape, lstm_units=50, mlp_units=[64], dropout=0.2):
    inputs = Input(shape=input_shape)
    x = LSTM(lstm_units, return_sequences=True)(inputs)
    x = Dropout(dropout)(x)
    x = LSTM(lstm_units)(x)
    x = Dropout(dropout)(x)
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(dropout)(x)
    outputs = Dense(3, activation="softmax")(x)
    return Model(inputs, outputs)

# --- 4. TFLite Conversion ---
def convert_to_tflite(model, model_name, save_dir="models/"):
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    file_path = os.path.join(save_dir, f"{model_name}.tflite")
    with open(file_path, "wb") as f: f.write(tflite_model)
    print(f"Successfully saved {model_name}.tflite")
    return file_path

# --- 5. Main Training Orchestrator ---
def main():
    version_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"--- Starting Training Run: {version_timestamp} ---")

    # --- USER TUNING: Core Training Parameters ---
    SYMBOL = "AAPL"
    SEQUENCE_LENGTH = 60      # How many minutes of data to look at for one prediction.
    TIME_HORIZON = 30         # How far into the future the label looks (vertical barrier).
    PT_SL_MULTIPLIERS = [1.5, 1.5] # [Profit_Take, Stop_Loss] multipliers of daily volatility.
    TRAIN_TEST_SPLIT_RATIO = 0.8 # 80% for training, 20% for testing.
    
    # --- Load Data ---
    print("Loading data...")
    # Production: Replace this with your database call.
    # db_manager = DatabaseManager()
    # raw_prices = db_manager.get_historical_data(symbol=SYMBOL, start_date="2022-01-01")
    # --- For Demonstration: Using dummy data ---
    dates = pd.to_datetime(pd.date_range("2025-01-01", periods=50000, freq="1min"))
    price_data = 150 + np.random.randn(50000).cumsum() * 0.1
    raw_prices = pd.DataFrame({'open': price_data, 'high': price_data + 0.1, 'low': price_data - 0.1, 'close': price_data, 'volume': np.random.randint(1000, 10000, 50000)}, index=dates)

    # --- Feature Engineering & Labeling ---
    print("Creating features and labels...")
    features_df, scaler_state = create_features(raw_prices, fit_scaler=True)
    labels_df = get_labels(features_df['close'], time_horizon=TIME_HORIZON, pt_sl_multipliers=PT_SL_MULTIPLIERS)
    aligned_df = features_df.join(labels_df, how='inner')
    
    # --- Prepare Sequences ---
    print("Preparing data sequences...")
    feature_cols = scaler_state['features']
    X = aligned_df[feature_cols]
    y = aligned_df['label']
    X_seq, y_seq = create_sequences(X, y, sequence_length=SEQUENCE_LENGTH)
    y_cat = to_categorical(y_seq, num_classes=3)

    # --- Chronological Split ---
    split_index = int(len(X_seq) * TRAIN_TEST_SPLIT_RATIO)
    X_train, X_test = X_seq[:split_index], X_seq[split_index:]
    y_train, y_test = y_cat[:split_index], y_cat[split_index:]
    print(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")
    
    # --- Save Training Stats for Drift Monitoring ---
    num_samples, seq_len, num_features = X_train.shape
    X_train_2d = X_train.reshape(num_samples * seq_len, num_features)
    training_df_for_stats = pd.DataFrame(X_train_2d, columns=feature_cols)
    training_stats = {"mean": training_df_for_stats.mean().to_dict(), "std": training_df_for_stats.std().to_dict()}
    stats_path = f"models/training_stats_{version_timestamp}.json"
    if not os.path.exists("models/"): os.makedirs("models/")
    with open(stats_path, 'w') as f: json.dump(training_stats, f, indent=4)
    print(f"Training stats saved to {stats_path}")

    # --- Train Models ---
    input_shape = (X_train.shape[1], X_train.shape[2])
    models_to_train = {
        "lstm": build_lstm_model(input_shape),
        "transformer": build_transformer_model(input_shape, head_size=256, num_heads=4, ff_dim=4, num_blocks=4, mlp_units=[128])
    }
    
    for name, model in models_to_train.items():
        print(f"\n--- Training {name.upper()} Model ---")
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        checkpoint_path = f"models/best_{name}_{version_timestamp}.h5"
        callbacks = [
            ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_accuracy', mode='max'),
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        ]
        model.fit(X_train, y_train, batch_size=64, epochs=100, validation_data=(X_test, y_test), callbacks=callbacks)
        model.load_weights(checkpoint_path)
        convert_to_tflite(model, f"{name}_{version_timestamp}")

    # --- Save Challenger Info for Validation Script ---
    scaler_path = f"models/scaler_state_{version_timestamp}.json"
    with open(scaler_path, 'w') as f: json.dump(scaler_state, f)
    challenger_info = {
        "lstm_model_path": f"models/lstm_{version_timestamp}.tflite",
        "transformer_model_path": f"models/transformer_{version_timestamp}.tflite",
        "scaler_path": scaler_path,
        "stats_path": stats_path
    }
    with open("models/challenger_info.json", 'w') as f: json.dump(challenger_info, f, indent=4)
    print("\nChallenger info saved for validation script.")

if __name__ == "__main__":
    main()