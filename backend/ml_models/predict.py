import json
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, Any, List

# Import from your project structure
from ml_models.feature_engineering import create_features

class ModelPredictor:
    """
    A class to load a trained TFLite model and its associated scaler
    to make predictions on new data.
    """
    def __init__(self, model_path: str, scaler_state_path: str):
        """
        Initializes the predictor by loading the TFLite model and scaler state.
        
        Args:
            model_path (str): Path to the .tflite model file.
            scaler_state_path (str): Path to the scaler_state.json file.
        """
        # Load the TFLite model and allocate tensors
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # Get input and output tensor details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Load the scaler state
        with open(scaler_state_path, 'r') as f:
            self.scaler_state = json.load(f)
            
        # Define the mapping from model output index to label
        self.label_map = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}
        
        # Extract sequence length from model's input shape
        self.sequence_length = self.input_details[0]['shape'][1]


    def predict(self, recent_price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Makes a prediction on a DataFrame of recent price data.

        Args:
            recent_price_data (pd.DataFrame): A DataFrame containing at least
                `sequence_length` rows of recent price data with columns
                ['open', 'high', 'low', 'close', 'volume'] and a datetime index.

        Returns:
            A dictionary containing the predicted label and class probabilities.
        """
        if len(recent_price_data) < self.sequence_length:
            raise ValueError(
                f"Input data must have at least {self.sequence_length} rows, "
                f"but got {len(recent_price_data)}."
            )

        # 1. Take the most recent `sequence_length` data points
        data_sequence = recent_price_data.tail(self.sequence_length)

        # 2. Create features using the saved scaler state
        # IMPORTANT: fit_scaler is False
        features_df, _ = create_features(
            prices=data_sequence, 
            fit_scaler=False, 
            scaler_state=self.scaler_state
        )

        # 3. Reshape the data to match the model's expected input shape
        # The shape should be (1, sequence_length, num_features)
        input_data = features_df[self.scaler_state['features']].values
        input_data = np.expand_dims(input_data, axis=0).astype(np.float32)

        # 4. Set the input tensor, invoke the interpreter, and get the output
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # 5. Interpret the output
        probabilities = output_data[0]
        predicted_index = np.argmax(probabilities)
        predicted_label = self.label_map[predicted_index]

        return {
            "prediction": predicted_label,
            "confidence": {
                "DOWN": float(probabilities[0]),
                "NEUTRAL": float(probabilities[1]),
                "UP": float(probabilities[2])
            }
        }

# Example Usage (for testing)
if __name__ == '__main__':
    # This assumes you have run train_model.py and have these files
    MODEL_PATH = 'models/transformer_v1.tflite'
    SCALER_PATH = 'models/scaler_state.json'
    
    # Create dummy data for the last 61 minutes for testing
    # In a real scenario, this would come from your database
    dates = pd.to_datetime(pd.date_range("2025-09-07 09:45:00", periods=61, freq="1min"))
    price_data = 150 + np.random.randn(61).cumsum() * 0.1
    dummy_prices = pd.DataFrame({
        'open': price_data, 'high': price_data + 0.1, 
        'low': price_data - 0.1, 'close': price_data, 
        'volume': np.random.randint(1000, 10000, 61)
    }, index=dates)

    try:
        predictor = ModelPredictor(model_path=MODEL_PATH, scaler_state_path=SCALER_PATH)
        prediction_result = predictor.predict(dummy_prices)
        print("Prediction successful!")
        print(json.dumps(prediction_result, indent=2))
    except FileNotFoundError:
        print("Error: Model or scaler file not found.")
        print("Please run train_model.py first to generate the necessary files.")