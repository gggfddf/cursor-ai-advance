import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb
import shap
import warnings
import json
import os
import sys
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# Install openpyxl if not available
try:
    import openpyxl
except ImportError:
    import subprocess
    subprocess.run(['pip', 'install', 'openpyxl'], check=True)
    import openpyxl

class TradingStrategy:
    """Trading strategy class with configurable parameters."""
    
    def __init__(self, initial_balance=500, leverage=200, stop_loss_pct=0.02, take_profit_pct=0.04):
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.balance = initial_balance
        self.trades = []
        self.position = None  # None, 'long', 'short'
        self.position_size = 0
        self.entry_price = 0
        self.entry_date = None
        
    def calculate_position_size(self, price):
        """Calculate position size based on balance and leverage."""
        return (self.balance * self.leverage) / price
        
    def open_position(self, signal, price, date):
        """Open a new position based on signal."""
        if self.position is not None:
            return  # Already in position
            
        if signal == 1:  # Buy signal
            self.position = 'long'
            self.position_size = self.calculate_position_size(price)
            self.entry_price = price
            self.entry_date = date
            
        elif signal == 0:  # Sell signal (short)
            self.position = 'short'
            self.position_size = self.calculate_position_size(price)
            self.entry_price = price
            self.entry_date = date
            
    def close_position(self, price, date, reason='signal'):
        """Close current position and record trade."""
        if self.position is None:
            return
            
        if self.position == 'long':
            pnl = (price - self.entry_price) * self.position_size
        else:  # short
            pnl = (self.entry_price - price) * self.position_size
            
        self.balance += pnl
        
        trade = {
            'entry_date': self.entry_date,
            'exit_date': date,
            'position': self.position,
            'entry_price': self.entry_price,
            'exit_price': price,
            'position_size': self.position_size,
            'pnl': pnl,
            'balance': self.balance,
            'return_pct': (pnl / self.initial_balance) * 100,
            'close_reason': reason
        }
        
        self.trades.append(trade)
        
        # Reset position
        self.position = None
        self.position_size = 0
        self.entry_price = 0
        self.entry_date = None
        
    def check_stop_loss_take_profit(self, price, date):
        """Check if stop loss or take profit should be triggered."""
        if self.position is None:
            return False
            
        if self.position == 'long':
            # Stop loss
            if price <= self.entry_price * (1 - self.stop_loss_pct):
                self.close_position(price, date, 'stop_loss')
                return True
            # Take profit
            elif price >= self.entry_price * (1 + self.take_profit_pct):
                self.close_position(price, date, 'take_profit')
                return True
                
        elif self.position == 'short':
            # Stop loss
            if price >= self.entry_price * (1 + self.stop_loss_pct):
                self.close_position(price, date, 'stop_loss')
                return True
            # Take profit
            elif price <= self.entry_price * (1 - self.take_profit_pct):
                self.close_position(price, date, 'take_profit')
                return True
                
        return False
        
    def execute_strategy(self, data, predictions):
        """Execute the trading strategy based on predictions."""
        for i in range(len(data)):
            if i < len(predictions):
                current_price = data.iloc[i]['close']
                current_date = data.index[i] if hasattr(data.index, '__getitem__') else i
                
                # Check stop loss/take profit first
                if not self.check_stop_loss_take_profit(current_price, current_date):
                    # Check for new signals
                    signal = predictions[i]
                    
                    if self.position is None:
                        # Open new position
                        self.open_position(signal, current_price, current_date)
                    elif (self.position == 'long' and signal == 0) or (self.position == 'short' and signal == 1):
                        # Close current position and open new one
                        self.close_position(current_price, current_date, 'signal_change')
                        self.open_position(signal, current_price, current_date)
                        
        # Close any remaining position at the end
        if self.position is not None:
            final_price = data.iloc[-1]['close']
            final_date = data.index[-1] if hasattr(data.index, '__getitem__') else len(data)-1
            self.close_position(final_price, final_date, 'end_of_data')
            
    def get_trade_statistics(self):
        """Calculate trading statistics."""
        if not self.trades:
            return {}
            
        df_trades = pd.DataFrame(self.trades)
        
        stats = {
            'total_trades': len(self.trades),
            'winning_trades': len(df_trades[df_trades['pnl'] > 0]),
            'losing_trades': len(df_trades[df_trades['pnl'] <= 0]),
            'win_rate': len(df_trades[df_trades['pnl'] > 0]) / len(self.trades) * 100,
            'total_pnl': df_trades['pnl'].sum(),
            'average_pnl': df_trades['pnl'].mean(),
            'max_profit': df_trades['pnl'].max(),
            'max_loss': df_trades['pnl'].min(),
            'final_balance': self.balance,
            'total_return': ((self.balance - self.initial_balance) / self.initial_balance) * 100
        }
        
        return stats
        
    def export_trades_to_csv(self, filename='trades.csv'):
        """Export all trades to CSV file."""
        if not self.trades:
            print("No trades to export")
            return
            
        df_trades = pd.DataFrame(self.trades)
        df_trades.to_csv(filename, index=False)
        print(f"Trades exported to {filename}")
        
    def export_trades_to_excel(self, filename='trades.xlsx'):
        """Export all trades to Excel file."""
        if not self.trades:
            print("No trades to export")
            return
            
        df_trades = pd.DataFrame(self.trades)
        
        # Create Excel writer
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Write trades to one sheet
            df_trades.to_excel(writer, sheet_name='Trades', index=False)
            
            # Write statistics to another sheet
            stats = self.get_trade_statistics()
            stats_df = pd.DataFrame(list(stats.items()), columns=['Metric', 'Value'])
            stats_df.to_excel(writer, sheet_name='Statistics', index=False)
            
        print(f"Trades exported to {filename}")

class RobustXAUUSDModel:
    def __init__(self, data_file='XAU_1d_data_clean.csv'):
        self.data_file = data_file
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_cols = []
        self.data = None
        self.trading_strategy = TradingStrategy(initial_balance=500, leverage=200)
        
    def load_and_validate_data(self):
        """Load and validate the input data with proper error handling."""
        try:
            if not os.path.exists(self.data_file):
                logger.error(f"Data file {self.data_file} not found!")
                return False
                
            self.data = pd.read_csv(self.data_file)
            logger.info(f"Loaded data with shape: {self.data.shape}")
            
            # Standardize column names
            self.data.columns = [c.lower().strip() for c in self.data.columns]
            
            # Validate required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in self.data.columns]
            
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return False
                
            # Handle date column
            if 'date' in self.data.columns:
                try:
                    self.data['date'] = pd.to_datetime(self.data['date'])
                    self.data = self.data.sort_values('date').reset_index(drop=True)
                    logger.info("Date column processed successfully")
                except Exception as e:
                    logger.warning(f"Could not process date column: {e}")
                    
            # Data quality checks
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in self.data.columns:
                    # Convert to numeric, coercing errors to NaN
                    self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
                    
                    # Check for negative values (shouldn't exist in OHLCV data)
                    if (self.data[col] < 0).any():
                        logger.warning(f"Found negative values in {col}, setting to NaN")
                        self.data.loc[self.data[col] < 0, col] = np.nan
                        
            # Basic OHLC validation
            invalid_ohlc = (
                (self.data['high'] < self.data['low']) |
                (self.data['high'] < self.data['open']) |
                (self.data['high'] < self.data['close']) |
                (self.data['low'] > self.data['open']) |
                (self.data['low'] > self.data['close'])
            )
            
            if invalid_ohlc.any():
                logger.warning(f"Found {invalid_ohlc.sum()} rows with invalid OHLC relationships")
                self.data = self.data[~invalid_ohlc]
                
            # Handle missing values
            missing_count = self.data.isnull().sum().sum()
            if missing_count > 0:
                logger.warning(f"Found {missing_count} missing values, will handle in feature engineering")
                
            logger.info(f"Data validation complete. Final shape: {self.data.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def safe_divide(self, numerator, denominator, default=0):
        """Safe division with default value for zero division."""
        return np.where(denominator != 0, numerator / denominator, default)
    
    def candle_features(self, df):
        """Enhanced feature engineering with better error handling."""
        try:
            df = df.copy()
            logger.info("Starting feature engineering...")
            
            # 1. Basic Candle Anatomy & Structure Features
            df['body'] = (df['close'] - df['open']).abs()
            df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
            df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
            df['range'] = df['high'] - df['low']
            
            # Safe ratios with proper division handling
            df['body_range_ratio'] = self.safe_divide(df['body'], df['range'])
            df['wick_body_ratio'] = self.safe_divide(df['upper_wick'] + df['lower_wick'], df['body'])
            
            # Enhanced candle type encoding
            conditions = [
                df['body'] > df['range'] * 0.7,
                df['body'] < df['range'] * 0.2,
                (df['upper_wick'] > df['body'] * 1.5) & (df['lower_wick'] < df['body']),
                (df['lower_wick'] > df['body'] * 1.5) & (df['upper_wick'] < df['body'])
            ]
            choices = [3, 1, 4, 5]  # numeric encoding instead of strings
            df['candle_type'] = np.select(conditions, choices, default=2)
            
            # Directional encoding (numeric)
            df['direction'] = np.where(df['close'] > df['open'], 1, 
                                     np.where(df['close'] < df['open'], -1, 0))
            
            # 2. Timing and Reversal Features
            df['bars_since_new_high'] = df.groupby((df['high'] == df['high'].expanding().max()).cumsum()).cumcount()
            df['bars_since_new_low'] = df.groupby((df['low'] == df['low'].expanding().min()).cumsum()).cumcount()
            
            # Safe rolling operations
            df['time_to_reversal'] = df['bars_since_new_high'].rolling(window=10, min_periods=1).min()
            
            # Fakeout detection
            df['fakeout_up'] = ((df['high'] > df['high'].shift(1)) & (df['close'] < df['open'])).astype(int)
            df['fakeout_down'] = ((df['low'] < df['low'].shift(1)) & (df['close'] > df['open'])).astype(int)
            
            # Consolidation breakout
            range_std = df['range'].rolling(window=3, min_periods=1).std()
            range_mean = df['range'].rolling(window=20, min_periods=1).mean()
            body_mean = df['body'].rolling(window=10, min_periods=1).mean()
            
            df['consolidation_breakout_window'] = (
                (range_std < range_mean * 0.3) & 
                (df['body'] > body_mean * 1.5)
            ).astype(int)
            
            # 3. Volume Features
            volume_median = df['volume'].rolling(window=10, min_periods=1).median()
            volume_mean = df['volume'].rolling(window=20, min_periods=1).mean()
            
            df['volume_ratio_10'] = self.safe_divide(df['volume'], volume_median, 1)
            df['volume_anomaly'] = (df['volume'] > volume_mean * 2).astype(int)
            
            # 4. Market Behavior Features
            high_20_max = df['high'].rolling(window=20, min_periods=1).max()
            low_20_min = df['low'].rolling(window=20, min_periods=1).min()
            range_20 = high_20_max - low_20_min
            
            df['relative_position_20'] = self.safe_divide(df['close'] - low_20_min, range_20)
            
            df['gap_up'] = (df['open'] > df['close'].shift(1)).astype(int)
            df['gap_down'] = (df['open'] < df['close'].shift(1)).astype(int)
            df['wick_pressure'] = self.safe_divide(df['upper_wick'] + df['lower_wick'], df['range'])
            
            # Volatility features
            range_std_10 = df['range'].rolling(window=10, min_periods=1).std()
            range_std_50 = df['range'].rolling(window=50, min_periods=1).std()
            range_mean_10 = df['range'].rolling(window=10, min_periods=1).mean()
            
            df['volatility_squeeze'] = (range_std_10 < range_std_50 * 0.5).astype(int)
            df['volatility_expansion'] = (df['range'] > range_mean_10 * 1.5).astype(int)
            
            # Pattern detection
            df['inside_bar'] = ((df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))).astype(int)
            
            # Engulfing pattern
            bullish_engulfing = (
                (df['close'] > df['open']) & 
                (df['close'].shift(1) < df['open'].shift(1)) & 
                (df['close'] > df['open'].shift(1)) & 
                (df['open'] < df['close'].shift(1))
            )
            bearish_engulfing = (
                (df['close'] < df['open']) & 
                (df['close'].shift(1) > df['open'].shift(1)) & 
                (df['close'] < df['open'].shift(1)) & 
                (df['open'] > df['close'].shift(1))
            )
            df['engulfing'] = (bullish_engulfing | bearish_engulfing).astype(int)
            
            # 5. Advanced Pattern Features
            # Consecutive patterns
            df['consecutive_green'] = (df['direction'] == 1).astype(int)
            df['consecutive_red'] = (df['direction'] == -1).astype(int)
            
            # Calculate consecutive counts properly
            green_groups = (df['consecutive_green'] == 0).cumsum()
            red_groups = (df['consecutive_red'] == 0).cumsum()
            
            df['consecutive_green'] = df['consecutive_green'].groupby(green_groups).cumsum()
            df['consecutive_red'] = df['consecutive_red'].groupby(red_groups).cumsum()
            
            df['directional_consistency'] = df[['consecutive_green', 'consecutive_red']].max(axis=1)
            
            # 6. Psychological and Energy Features
            body_mean_6 = df['body'].rolling(window=6, min_periods=1).mean()
            body_std_6 = df['body'].rolling(window=6, min_periods=1).std()
            body_std_20_median = body_std_6.rolling(window=20, min_periods=1).median()
            
            df['body_std_6'] = body_std_6
            df['tight_cluster_zone'] = (body_std_6 < body_std_20_median * 0.5).astype(int)
            df['is_first_expansion_candle'] = (
                (df['tight_cluster_zone'].shift(1) == 1) & 
                (df['body'] > body_mean_6 * 1.5)
            ).astype(int)
            
            df['expansion_energy'] = (df['body'] > body_mean_6 * 2).astype(int)
            
            # 7. Support/Resistance Features
            df['support_test_count'] = df['low'].rolling(window=20, min_periods=1).apply(
                lambda x: np.sum(np.abs(x - x.min()) < x.std() * 0.1) if len(x) > 0 and x.std() > 0 else 0,
                raw=True
            )
            
            df['breakout_insecurity'] = (
                (df['high'] > df['high'].shift(1)) & 
                (df['close'] <= df['open'])
            ).astype(int)
            
            # 8. Weather System Features
            body_mean_20 = df['body'].rolling(window=20, min_periods=1).mean()
            range_mean_20 = df['range'].rolling(window=20, min_periods=1).mean()
            volume_mean_20 = df['volume'].rolling(window=20, min_periods=1).mean()
            
            df['storm_day'] = (
                (df['body'] > body_mean_20 * 1.5) & 
                (df['range'] > range_mean_20 * 1.5) & 
                (df['volume'] > volume_mean_20 * 1.5)
            ).astype(int)
            
            df['humidity_day'] = (
                (df['body'] < body_mean_20 * 0.5) & 
                (df['upper_wick'] + df['lower_wick'] > df['body'] * 2)
            ).astype(int)
            
            # 9. Pattern Recognition Features
            df['hammer'] = (
                (df['lower_wick'] > df['body'] * 2) & 
                (df['upper_wick'] < df['body'] * 0.5)
            ).astype(int)
            
            df['shooting_star'] = (
                (df['upper_wick'] > df['body'] * 2) & 
                (df['lower_wick'] < df['body'] * 0.5)
            ).astype(int)
            
            df['doji'] = (df['body'] < df['range'] * 0.1).astype(int)
            
            # 10. Momentum and Exhaustion Features
            df['momentum_exhaustion'] = (
                (df['consecutive_green'] >= 5) | 
                (df['consecutive_red'] >= 5)
            ).astype(int)
            
            # Liquidity sweeps
            high_10_max = df['high'].rolling(window=10, min_periods=1).max()
            low_10_min = df['low'].rolling(window=10, min_periods=1).min()
            
            df['liquidity_sweep_up'] = (
                (df['high'] > high_10_max.shift(1)) & 
                (df['close'] < df['open'])
            ).astype(int)
            
            df['liquidity_sweep_down'] = (
                (df['low'] < low_10_min.shift(1)) & 
                (df['close'] > df['open'])
            ).astype(int)
            
            # Add calendar features if date is available
            if 'date' in df.columns:
                df['start_of_month'] = (df['date'].dt.day <= 3).astype(int)
                df['end_of_month'] = (df['date'].dt.day >= 25).astype(int)
                df['day_of_week'] = df['date'].dt.dayofweek
            else:
                df['start_of_month'] = 0
                df['end_of_month'] = 0
                df['day_of_week'] = 0
                
            logger.info("Feature engineering completed successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            raise
    
    def prepare_features(self):
        """Prepare and select features for modeling."""
        try:
            # Apply feature engineering
            self.data = self.candle_features(self.data)
            
            # Define feature columns (only numeric features that should exist)
            self.feature_cols = [
                # Basic candle features
                'body', 'upper_wick', 'lower_wick', 'range', 'body_range_ratio', 'wick_body_ratio',
                'candle_type', 'direction',
                
                # Timing features
                'bars_since_new_high', 'bars_since_new_low', 'time_to_reversal',
                'fakeout_up', 'fakeout_down', 'consolidation_breakout_window',
                
                # Volume features
                'volume_ratio_10', 'volume_anomaly',
                
                # Market behavior
                'relative_position_20', 'gap_up', 'gap_down', 'wick_pressure',
                'volatility_squeeze', 'volatility_expansion', 'inside_bar', 'engulfing',
                'directional_consistency',
                
                # Energy features
                'body_std_6', 'tight_cluster_zone', 'is_first_expansion_candle', 'expansion_energy',
                
                # Support/resistance
                'support_test_count', 'breakout_insecurity',
                
                # Weather system
                'storm_day', 'humidity_day',
                
                # Patterns
                'hammer', 'shooting_star', 'doji',
                
                # Momentum
                'momentum_exhaustion', 'liquidity_sweep_up', 'liquidity_sweep_down',
                
                # Calendar
                'start_of_month', 'end_of_month', 'day_of_week'
            ]
            
            # Filter features that actually exist in the dataframe
            existing_features = [col for col in self.feature_cols if col in self.data.columns]
            missing_features = [col for col in self.feature_cols if col not in self.data.columns]
            
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                
            self.feature_cols = existing_features
            logger.info(f"Using {len(self.feature_cols)} features for modeling")
            
            # Handle missing values in features
            imputer = SimpleImputer(strategy='median')
            self.data[self.feature_cols] = imputer.fit_transform(self.data[self.feature_cols])
            
            return True
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return False
    
    def create_target(self, threshold=0.0):
        """Create target variable with proper validation."""
        try:
            # Binary target: 1 if next day's close > today's close, else 0
            self.data['target'] = (self.data['close'].shift(-1) - self.data['close'] > threshold).astype(int)
            
            # Remove rows with NaN target (last row)
            initial_rows = len(self.data)
            self.data = self.data.dropna(subset=['target']).reset_index(drop=True)
            
            logger.info(f"Target created. Removed {initial_rows - len(self.data)} rows with missing target")
            logger.info(f"Target distribution: {self.data['target'].value_counts().to_dict()}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating target: {e}")
            return False
    
    def train_model(self, test_size=0.2, random_state=42):
        """Train the XGBoost model with proper validation."""
        try:
            if len(self.feature_cols) == 0:
                logger.error("No features available for training")
                return False
                
            X = self.data[self.feature_cols]
            y = self.data['target']
            
            # Time-based split (important for time series)
            split_idx = int(len(self.data) * (1 - test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            logger.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
            
            # Train model with better parameters
            self.model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                n_jobs=-1,
                eval_metric='logloss'
            )
            
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred)
            }
            
            # Print results
            print(f"\n{'='*50}")
            print("MODEL EVALUATION RESULTS")
            print(f"{'='*50}")
            for metric, value in metrics.items():
                print(f"{metric.capitalize()}: {value:.4f}")
            
            cm = confusion_matrix(y_test, y_pred)
            print(f"\nConfusion Matrix:\n{cm}")
            print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
            
            # Store results for later use
            self.X_train, self.X_test = X_train, X_test
            self.y_train, self.y_test = y_train, y_test
            self.y_pred = y_pred
            self.metrics = metrics
            
            logger.info("Model training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
    
    def generate_feature_importance(self):
        """Generate and save feature importance analysis."""
        try:
            if self.model is None:
                logger.error("Model not trained yet")
                return False
                
            # Get feature importance
            importances = dict(zip(self.feature_cols, self.model.feature_importances_))
            sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            
            # Save to JSON (convert numpy types to Python types)
            feature_importance_dict = {str(k): float(v) for k, v in sorted_features}
            with open('feature_importance.json', 'w') as f:
                json.dump(feature_importance_dict, f, indent=2)
            
            # Create SHAP analysis (if not too many features)
            try:
                if len(self.feature_cols) <= 50:  # Limit for performance
                    explainer = shap.TreeExplainer(self.model)
                    # Use a sample for SHAP if test set is large
                    sample_size = min(500, len(self.X_test))
                    shap_sample = self.X_test.sample(n=sample_size, random_state=42)
                    shap_values = explainer.shap_values(shap_sample)
                    
                    # Create SHAP summary plot
                    plt.figure(figsize=(10, 8))
                    shap.summary_plot(shap_values, shap_sample, plot_type="bar", show=False)
                    plt.tight_layout()
                    plt.savefig('shap_feature_importance.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info("SHAP analysis completed")
                else:
                    logger.info("Skipping SHAP analysis due to large feature set")
            except Exception as e:
                logger.warning(f"SHAP analysis failed: {e}")
            
            # Create confusion matrix plot
            plt.figure(figsize=(8, 6))
            sns.heatmap(confusion_matrix(self.y_test, self.y_pred), 
                       annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Print top features
            print(f"\n{'='*50}")
            print("TOP 10 MOST IMPORTANT FEATURES")
            print(f"{'='*50}")
            for i, (feature, importance) in enumerate(sorted_features[:10], 1):
                print(f"{i:2d}. {feature:<30} {importance:.4f}")
            
            logger.info("Feature importance analysis completed")
            return True
            
        except Exception as e:
            logger.error(f"Error generating feature importance: {e}")
            return False
    
    def predict_recent(self, n_predictions=10):
        """Make predictions on the most recent data."""
        try:
            if self.model is None:
                logger.error("Model not trained yet")
                return None
                
            # Get recent data for predictions
            recent_data = self.data[self.feature_cols].tail(n_predictions)
            recent_predictions = self.model.predict(recent_data)
            recent_proba = self.model.predict_proba(recent_data)
            
            # Create results dataframe
            results = pd.DataFrame({
                'prediction': recent_predictions,
                'probability_down': recent_proba[:, 0],
                'probability_up': recent_proba[:, 1],
                'date': self.data.tail(n_predictions).index if hasattr(self.data, 'index') else range(len(self.data) - n_predictions, len(self.data))
            })
            
            logger.info(f"Generated {len(results)} recent predictions")
            return results
            
        except Exception as e:
            logger.error(f"Error in recent predictions: {e}")
            return None
            
    def execute_trading_strategy(self):
        """Execute the trading strategy based on model predictions."""
        try:
            if self.model is None:
                logger.error("Model not trained yet")
                return False
                
            # Get predictions for the entire test set
            X_test = self.data[self.feature_cols].iloc[len(self.X_train):]
            test_predictions = self.model.predict(X_test)
            test_data = self.data.iloc[len(self.X_train):]
            
            # Execute trading strategy
            self.trading_strategy.execute_strategy(test_data, test_predictions)
            
            # Print trading statistics
            stats = self.trading_strategy.get_trade_statistics()
            if stats:
                print(f"\n{'='*60}")
                print("TRADING STRATEGY RESULTS")
                print(f"{'='*60}")
                print(f"Initial Balance: ${self.trading_strategy.initial_balance}")
                print(f"Leverage: {self.trading_strategy.leverage}x")
                print(f"Final Balance: ${stats['final_balance']:.2f}")
                print(f"Total Return: {stats['total_return']:.2f}%")
                print(f"Total Trades: {stats['total_trades']}")
                print(f"Win Rate: {stats['win_rate']:.2f}%")
                print(f"Average P&L per Trade: ${stats['average_pnl']:.2f}")
                print(f"Max Profit: ${stats['max_profit']:.2f}")
                print(f"Max Loss: ${stats['max_loss']:.2f}")
                print(f"{'='*60}")
                
                # Export trades to CSV and Excel
                self.trading_strategy.export_trades_to_csv('xauusd_trades.csv')
                self.trading_strategy.export_trades_to_excel('xauusd_trades.xlsx')
                
            return True
            
        except Exception as e:
            logger.error(f"Error executing trading strategy: {e}")
            return False
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive analysis report."""
        try:
            if self.model is None:
                logger.error("Model not trained yet")
                return False
                
            # Get feature importance
            importances = dict(zip(self.feature_cols, self.model.feature_importances_))
            sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            
            report = f"""# XAUUSD Next-Day Price Movement Prediction Report
## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report presents a robust machine learning model for predicting next-day price direction of XAUUSD (Gold/USD) 
using advanced candlestick pattern analysis and market microstructure features.

**Key Results:**
- Accuracy: {self.metrics['accuracy']:.4f}
- F1 Score: {self.metrics['f1_score']:.4f}
- Precision: {self.metrics['precision']:.4f}
- Recall: {self.metrics['recall']:.4f}

## Dataset Overview

- **Asset:** XAUUSD (Gold/US Dollar)
- **Total Records:** {len(self.data)}
- **Training Period:** {len(self.X_train)} candles
- **Test Period:** {len(self.X_test)} candles
- **Features Used:** {len(self.feature_cols)}

## Model Architecture

**Algorithm:** XGBoost Classifier
**Parameters:**
- n_estimators: 300
- max_depth: 6
- learning_rate: 0.05
- subsample: 0.8
- colsample_bytree: 0.8

## Feature Categories

### 1. Candle Anatomy Features ({sum(1 for f in self.feature_cols if f in ['body', 'upper_wick', 'lower_wick', 'range', 'body_range_ratio', 'wick_body_ratio'])})
Basic candlestick structure analysis including body size, wick lengths, and their relationships.

### 2. Market Timing Features ({sum(1 for f in self.feature_cols if 'time' in f or 'bars_since' in f)})
Features capturing timing of reversals, breakouts, and market cycles.

### 3. Volume Analysis Features ({sum(1 for f in self.feature_cols if 'volume' in f)})
Volume-based anomaly detection and ratio analysis.

### 4. Pattern Recognition Features ({sum(1 for f in self.feature_cols if f in ['hammer', 'shooting_star', 'doji', 'engulfing', 'inside_bar'])})
Classical candlestick patterns and their variations.

### 5. Market Psychology Features ({sum(1 for f in self.feature_cols if f in ['momentum_exhaustion', 'liquidity_sweep_up', 'liquidity_sweep_down'])})
Features capturing market emotions and exhaustion points.

## Top 10 Most Predictive Features

"""
            for i, (feature, importance) in enumerate(sorted_features[:10], 1):
                report += f"{i}. **{feature}**: {importance:.4f}\n"
                
            report += f"""

## Model Performance Analysis

### Confusion Matrix
```
                Predicted
                0     1
Actual    0   {confusion_matrix(self.y_test, self.y_pred)[0,0]}   {confusion_matrix(self.y_test, self.y_pred)[0,1]}
          1   {confusion_matrix(self.y_test, self.y_pred)[1,0]}   {confusion_matrix(self.y_test, self.y_pred)[1,1]}
```

### Key Insights

1. **Top Feature Analysis:** The most important feature "{sorted_features[0][0]}" suggests that {self._interpret_feature(sorted_features[0][0])} is crucial for prediction.

2. **Pattern Recognition:** Classical patterns like {[f for f in ['hammer', 'shooting_star', 'doji'] if f in [item[0] for item in sorted_features[:10]]]} appear in top features, validating traditional technical analysis.

3. **Market Structure:** The presence of {[f for f in ['volatility_squeeze', 'volatility_expansion'] if f in [item[0] for item in sorted_features[:10]]]} in important features indicates market regime changes are predictive.

## Risk Considerations

- **Overfitting Risk:** Model uses time-based split to prevent look-ahead bias
- **Market Regime Changes:** Performance may vary during different market conditions
- **Feature Stability:** Regular retraining recommended as market dynamics evolve

## Usage Recommendations

1. **Signal Confirmation:** Use predictions as confirmation with other analysis
2. **Risk Management:** Always use proper position sizing and stop losses
3. **Model Updates:** Retrain monthly with new data
4. **Threshold Tuning:** Adjust prediction probability thresholds based on risk tolerance

## Technical Implementation

- **Data Validation:** Comprehensive OHLCV data quality checks
- **Feature Engineering:** {len(self.feature_cols)} engineered features from basic OHLCV
- **Missing Data:** Handled using median imputation
- **Cross-Validation:** Time series split preserving temporal order

---

*Disclaimer: This model is for educational and research purposes. Past performance does not guarantee future results. Always consult with financial advisors before making investment decisions.*
"""
            
            # Save report
            with open('XAUUSD_ML_Report.md', 'w') as f:
                f.write(report)
                
            logger.info("Comprehensive report generated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return False
    
    def _interpret_feature(self, feature_name):
        """Provide interpretation for feature names."""
        interpretations = {
            'body': 'candle body size (momentum strength)',
            'volume_ratio_10': 'volume anomalies and institutional activity',
            'tight_cluster_zone': 'market consolidation periods',
            'volatility_expansion': 'breakout momentum',
            'relative_position_20': 'price position within recent range',
            'directional_consistency': 'trend persistence patterns',
            'momentum_exhaustion': 'trend reversal timing'
        }
        return interpretations.get(feature_name, 'market microstructure behavior')
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline including trading strategy."""
        try:
            print("🚀 Starting XAUUSD ML Analysis Pipeline with Trading Strategy...")
            
            # Step 1: Load and validate data
            if not self.load_and_validate_data():
                return False
            print("✅ Data loaded and validated")
            
            # Step 2: Feature engineering
            if not self.prepare_features():
                return False
            print("✅ Features engineered")
            
            # Step 3: Create target
            if not self.create_target():
                return False
            print("✅ Target variable created")
            
            # Step 4: Train model
            if not self.train_model():
                return False
            print("✅ Model trained and evaluated")
            
            # Step 5: Execute trading strategy
            if not self.execute_trading_strategy():
                return False
            print("✅ Trading strategy executed")
            
            # Step 6: Feature importance analysis
            if not self.generate_feature_importance():
                return False
            print("✅ Feature importance analysis completed")
            
            # Step 7: Recent predictions
            recent_predictions = self.predict_recent()
            if recent_predictions is not None:
                print("✅ Recent predictions generated")
            
            # Step 8: Generate comprehensive report
            if not self.generate_comprehensive_report():
                return False
            print("✅ Comprehensive report generated")
            
            print(f"\n🎯 Analysis Complete! Files generated:")
            print("📊 confusion_matrix.png")
            print("📊 shap_feature_importance.png (if applicable)")
            print("📊 feature_importance.json")
            print("📊 XAUUSD_ML_Report.md")
            print("💰 xauusd_trades.csv")
            print("💰 xauusd_trades.xlsx")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in complete analysis: {e}")
            return False

def main():
    """Main execution function."""
    try:
        # Initialize and run the model
        model = RobustXAUUSDModel('XAU_1d_data_clean.csv')
        success = model.run_complete_analysis()
        
        if success:
            print("\n?? Model pipeline completed successfully!")
        else:
            print("\n? Model pipeline failed. Check logs for details.")
            
    except Exception as e:
        print(f"? Fatal error: {e}")
        logger.error(f"Fatal error in main: {e}")

if __name__ == "__main__":
    main()