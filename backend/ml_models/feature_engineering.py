import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

def create_features(
    prices: pd.DataFrame, 
    fit_scaler: bool = True, 
    scaler_state: Optional[Dict[str, Any]] = None
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """
    Master function to create a rich feature set from raw price data.

    Args:
        prices (pd.DataFrame): DataFrame with ['open', 'high', 'low', 'close', 'volume'] and a datetime index.
        fit_scaler (bool): If True, fits a new scaler and returns its state. If False, uses the provided scaler_state.
        scaler_state (Optional[Dict]): The state of a previously fitted MinMaxScaler. Required if fit_scaler is False.

    Returns:
        A tuple containing:
        - pd.DataFrame: The DataFrame with all engineered features.
        - Optional[Dict]: The state of the scaler if fit_scaler was True, otherwise None.
    """
    df = prices.copy()
    
    # --- 1. Technical Indicators ---
    # Customize these indicators and their parameters based on your trading strategy.
    df["ema_12"] = ta.ema(df["close"], length=12)
    df["ema_26"] = ta.ema(df["close"], length=26)
    df["rsi_14"] = ta.rsi(df["close"], length=14)
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df = df.join(macd)
    df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    bbands = ta.bbands(df["close"], length=20)
    df = df.join(bbands)
    df["roc_10"] = ta.roc(df["close"], length=10)
    df["vwma_20"] = ta.vwma(df["close"], df["volume"], length=20)

    # --- 2. Time-Based Features ---
    df["minute_of_hour"] = df.index.minute
    df["hour_of_day"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["minute_sin"] = np.sin(2 * np.pi * df["minute_of_hour"] / 60)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute_of_hour"] / 60)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    
    # --- 3. Lagged Features ---
    # These features help the model understand recent price action.
    for lag in [1, 5, 10, 15]:
        df[f'log_return_{lag}m'] = np.log(df['close'] / df['close'].shift(lag))

    # --- 4. Final Processing ---
    # Drop rows with NaNs created by indicators/lags at the start of the series.
    df.dropna(inplace=True)

    if df.empty:
        raise ValueError("DataFrame is empty after feature creation and NaN removal. Ensure input data is long enough.")

    # --- 5. Scaling ---
    scaler = MinMaxScaler()
    # 'close' is excluded because it's our basis for labeling, not a direct feature for prediction.
    exclude_cols = ["open", "high", "low", "close", "symbol", "ts"] 
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    if fit_scaler:
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        new_scaler_state = {
            "min": scaler.min_.tolist(),
            "scale": scaler.scale_.tolist(),
            "features": feature_cols
        }
        return df, new_scaler_state
    else:
        if scaler_state is None:
            raise ValueError("scaler_state must be provided when fit_scaler is False.")
        
        # Ensure the columns in the dataframe align with the scaler's features for robust transformation.
        ordered_feature_cols = scaler_state["features"]
        df_to_scale = df[ordered_feature_cols]

        scaler.min_ = np.array(scaler_state["min"])
        scaler.scale_ = np.array(scaler_state["scale"])

        df[ordered_feature_cols] = scaler.transform(df_to_scale)
        return df, None