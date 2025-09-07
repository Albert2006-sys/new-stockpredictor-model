import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

# --- USER TUNING: Drift Parameters ---
MODEL_CONFIG_PATH = 'config/model_config.json'
# How many standard deviations away from the training mean constitutes a data drift.
DATA_DRIFT_THRESHOLD = 3.0 
# The minimum average confidence the model should have to be considered healthy.
CONCEPT_DRIFT_CONFIDENCE_THRESHOLD = 0.60 
# How far back to look for live data (e.g., last 60 minutes).
MONITORING_WINDOW_MINUTES = 60

class DriftMonitor:
    def __init__(self, config_path: str):
        print("Initializing Drift Monitor...")
        try:
            with open(config_path, 'r') as f: self.config = json.load(f)
            champion_info = self.config.get('current_champion')
            if not champion_info: raise ValueError("No 'current_champion' found in config.")
            with open(champion_info.get('stats_path'), 'r') as f: self.training_stats = json.load(f)
            self.feature_names = list(self.training_stats['mean'].keys())
            print("Drift Monitor initialized successfully.")
        except Exception as e:
            print(f"Error initializing Drift Monitor: {e}. A model must be trained and promoted first.")
            self.training_stats = None

    def fetch_recent_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        print(f"Fetching data from the last {MONITORING_WINDOW_MINUTES} minutes...")
        # Production: Replace this with queries to your database for prices and prediction logs.
        # --- For Demonstration: Simulating live data ---
        end_time = pd.Timestamp.now()
        start_time = end_time - pd.Timedelta(minutes=MONITORING_WINDOW_MINUTES)
        dates = pd.to_datetime(pd.date_range(start_time, end_time, freq="1min"))
        price_data = 170 + np.random.randn(len(dates)).cumsum() * 0.1
        # To test data drift, we'll occasionally generate out-of-distribution volume.
        volume_data = np.random.randint(50000, 100000, len(dates)) if np.random.rand() > 0.5 else np.random.randint(1000, 10000, len(dates))
        live_prices = pd.DataFrame({'open': price_data, 'high': price_data + 0.1, 'low': price_data - 0.1, 'close': price_data, 'volume': volume_data}, index=dates)
        
        # To test concept drift, we'll occasionally generate low-confidence predictions.
        if np.random.rand() > 0.5: probs = np.random.dirichlet(np.ones(3), size=len(dates)) # Healthy
        else: probs = np.random.dirichlet(np.ones(3) * 0.2, size=len(dates)) # Unhealthy (low confidence)
        prediction_logs = pd.DataFrame(probs, columns=['DOWN', 'NEUTRAL', 'UP'], index=dates)
        return live_prices, prediction_logs

    def check_data_drift(self, live_features: pd.DataFrame) -> Dict[str, Any]:
        report = {"is_drifted": False, "drifted_features": {}}
        live_mean = live_features[self.feature_names].mean()
        for feature in self.feature_names:
            mean, std = self.training_stats['mean'][feature], self.training_stats['std'][feature]
            z_score = abs((live_mean[feature] - mean) / (std + 1e-8))
            if z_score > DATA_DRIFT_THRESHOLD:
                report["is_drifted"] = True
                report["drifted_features"][feature] = {"live_mean": live_mean[feature], "training_mean": mean, "z_score": z_score}
        return report

    def check_concept_drift(self, prediction_logs: pd.DataFrame) -> Dict[str, Any]:
        average_confidence = prediction_logs[['DOWN', 'NEUTRAL', 'UP']].max(axis=1).mean()
        is_drifted = average_confidence < CONCEPT_DRIFT_CONFIDENCE_THRESHOLD
        return {"is_drifted": is_drifted, "average_confidence": average_confidence}

    def trigger_rollback(self) -> bool:
        print("\nCRITICAL: Severe drift detected. Attempting to roll back model.")
        if not self.config.get('previous_champion'):
            print("Rollback failed: No 'previous_champion' found in config.")
            return False
        self.config['current_champion'] = self.config['previous_champion']
        self.config['previous_champion'] = None
        try:
            with open(MODEL_CONFIG_PATH, 'w') as f: json.dump(self.config, f, indent=4)
            print("SUCCESS: Model rolled back to previous stable version.")
            return True
        except Exception as e:
            print(f"FATAL: Failed to write updated config during rollback: {e}")
            return False

    def run_check(self):
        if not self.training_stats: return
        from ml_models.feature_engineering import create_features

        live_prices, prediction_logs = self.fetch_recent_data()
        
        champion_info = self.config['current_champion']
        with open(champion_info['scaler_path'], 'r') as f: scaler_state = json.load(f)
        live_features, _ = create_features(live_prices, fit_scaler=False, scaler_state=scaler_state)
        
        data_drift_report = self.check_data_drift(live_features)
        print("\n--- Data Drift Report ---")
        if data_drift_report['is_drifted']:
            print("🚨 DATA DRIFT DETECTED!"); print(json.dumps(data_drift_report['drifted_features'], indent=2))
        else: print("✅ No data drift detected.")
            
        concept_drift_report = self.check_concept_drift(prediction_logs)
        print("\n--- Concept Drift Report ---")
        print(f"Average Model Confidence: {concept_drift_report['average_confidence']:.2%}")
        if concept_drift_report['is_drifted']:
            print(f"🚨 CONCEPT DRIFT DETECTED! Confidence is below the {CONCEPT_DRIFT_CONFIDENCE_THRESHOLD:.0%} threshold.")
        else: print("✅ Model confidence is healthy.")
            
        if data_drift_report['is_drifted'] or concept_drift_report['is_drifted']:
            self.trigger_rollback()

if __name__ == '__main__':
    monitor = DriftMonitor(config_path=MODEL_CONFIG_PATH)
    monitor.run_check()