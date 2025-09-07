import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report

# Import from your project structure
from ml_models.feature_engineering import create_features
from ml_models.train_model import get_labels, create_sequences

# --- USER TUNING: Validation Parameters ---
MODEL_CONFIG_PATH = 'config/model_config.json'
# A new model must be at least 2% better on the F1-score to be promoted.
# This prevents frequent model changes due to random statistical noise.
PROMOTION_THRESHOLD = 1.02 

def evaluate_model(model_path: str, X_val: np.ndarray, y_val_cat: np.ndarray) -> dict:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    predictions = []
    for x_sample in X_val:
        input_data = np.expand_dims(x_sample, axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(np.argmax(output_data[0]))
        
    y_true = np.argmax(y_val_cat, axis=1)
    
    print(f"\n--- Validation Report for {os.path.basename(model_path)} ---")
    report_str = classification_report(y_true, predictions, target_names=["DOWN", "NEUTRAL", "UP"])
    print(report_str)
    return classification_report(y_true, predictions, output_dict=True)

def main():
    print("--- Starting Model Validation ---")

    # 1. Load Champion model config
    try:
        with open(MODEL_CONFIG_PATH, 'r') as f: model_config = json.load(f)
        champion_info = model_config.get('current_champion', {})
    except FileNotFoundError:
        print(f"Warning: {MODEL_CONFIG_PATH} not found. Assuming no champion model exists.")
        model_config, champion_info = {}, {}

    # 2. Find Challenger model info
    try:
        with open("models/challenger_info.json", 'r') as f: challenger_info = json.load(f)
    except FileNotFoundError:
        print("Error: models/challenger_info.json not found. Run train_model.py first. Aborting.")
        return

    # 3. Load and prepare a dedicated validation dataset
    print("Loading and preparing validation data...")
    # Production: Replace this with a call to your database for the most recent data
    # that was NOT part of the training set (e.g., the last week).
    # --- For Demonstration: Using dummy data ---
    dates = pd.to_datetime(pd.date_range("2025-08-01", periods=10000, freq="1min"))
    price_data = 160 + np.random.randn(10000).cumsum() * 0.1
    val_prices = pd.DataFrame({'open': price_data, 'high': price_data + 0.1, 'low': price_data - 0.1, 'close': price_data, 'volume': np.random.randint(1000, 10000, 10000)}, index=dates)
    
    with open(challenger_info['scaler_path'], 'r') as f: scaler_state = json.load(f)
    
    val_features, _ = create_features(val_prices, fit_scaler=False, scaler_state=scaler_state)
    val_labels = get_labels(val_features['close'], time_horizon=30, pt_sl_multipliers=[1.5, 1.5])
    aligned_val = val_features.join(val_labels, how='inner')
    
    feature_cols = scaler_state['features']
    sequence_length = 60 # Should match training
    
    X_val, y_val_seq = create_sequences(aligned_val[feature_cols], aligned_val['label'], sequence_length)
    y_val_cat = tf.keras.utils.to_categorical(y_val_seq, num_classes=3)
    
    # 4. Evaluate both sets of models
    challenger_lstm_report = evaluate_model(challenger_info['lstm_model_path'], X_val, y_val_cat)
    challenger_transformer_report = evaluate_model(challenger_info['transformer_model_path'], X_val, y_val_cat)
    challenger_avg_f1 = (challenger_lstm_report['weighted avg']['f1-score'] + challenger_transformer_report['weighted avg']['f1-score']) / 2
    
    if champion_info:
        champion_lstm_report = evaluate_model(champion_info['lstm_model_path'], X_val, y_val_cat)
        champion_transformer_report = evaluate_model(champion_info['transformer_model_path'], X_val, y_val_cat)
        champion_avg_f1 = (champion_lstm_report['weighted avg']['f1-score'] + champion_transformer_report['weighted avg']['f1-score']) / 2
    else:
        champion_avg_f1 = 0 # No champion, challenger wins by default

    # 5. Compare and update config
    print("\n--- Final Comparison ---")
    print(f"Challenger Average F1-Score: {challenger_avg_f1:.4f}")
    print(f"Champion Average F1-Score:   {champion_avg_f1:.4f}")

    if challenger_avg_f1 > champion_avg_f1 * PROMOTION_THRESHOLD:
        print("\nOutcome: Challenger is better! Promoting to Champion. ✅")
        model_config['previous_champion'] = model_config.get('current_champion') # Old champion becomes fallback
        model_config['current_champion'] = challenger_info
    else:
        print("\nOutcome: Champion remains superior. No change. 🛡️")

    with open(MODEL_CONFIG_PATH, 'w') as f: json.dump(model_config, f, indent=4)
    print(f"Updated {MODEL_CONFIG_PATH}")

if __name__ == '__main__':
    main()