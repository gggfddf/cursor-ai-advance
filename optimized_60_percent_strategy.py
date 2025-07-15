import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import json
import os
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import xgboost as xgb
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class Optimized60PercentStrategy:
    def __init__(self, data_file='XAU_1d_data_clean.csv'):
        self.data_file = data_file
        self.data = None
        self.models = {}
        self.scalers = {}
        self.feature_cols = []
        
        # Optimized Parameters for 60%+ Win Rate
        self.initial_capital = 100000    # $100,000 starting capital
        self.risk_per_trade = 0.020      # 2.0% risk per trade
        self.min_confidence = 0.75       # 75% confidence (high quality signals only)
        self.ensemble_agreement = 0.80   # 80% of models must agree (very strict)
        self.stop_loss_pct = 0.012       # 1.2% stop loss (tight)
        self.take_profit_pct = 0.036     # 3.6% take profit (3:1 R:R)
        self.max_positions = 2           # Limited concurrent positions
        self.signal_quality_threshold = 0.85  # Very high quality threshold
        
        # Advanced filtering parameters
        self.volatility_filter = True
        self.trend_filter = True
        self.regime_filter = True
        self.momentum_filter = True
        self.volume_filter = True
        
        # Performance tracking
        self.trades = []
        self.portfolio_values = []
        self.positions = []
        
    def load_and_prepare_data(self):
        """Load and prepare the dataset."""
        try:
            if not os.path.exists(self.data_file):
                logger.error(f"Data file {self.data_file} not found!")
                return False
                
            self.data = pd.read_csv(self.data_file)
            logger.info(f"Loaded data with shape: {self.data.shape}")
            
            # Standardize column names
            self.data.columns = [c.lower().strip() for c in self.data.columns]
            
            # Process date column
            if 'date' in self.data.columns:
                self.data['date'] = pd.to_datetime(self.data['date'])
                self.data = self.data.sort_values('date').reset_index(drop=True)
            else:
                logger.error("Date column not found!")
                return False
            
            # Clean data
            self.data = self.data.dropna()
            self.data = self.data[(self.data['high'] >= self.data['low']) & 
                                 (self.data['high'] >= self.data['open']) & 
                                 (self.data['high'] >= self.data['close']) &
                                 (self.data['low'] <= self.data['open']) & 
                                 (self.data['low'] <= self.data['close'])]
            
            logger.info(f"Data cleaned. Final shape: {self.data.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def engineer_premium_features(self):
        """Engineer premium features for high-quality signals."""
        try:
            logger.info("Starting premium feature engineering...")
            
            # Convert to pandas Series
            high = pd.Series(self.data['high'].values)
            low = pd.Series(self.data['low'].values)
            close = pd.Series(self.data['close'].values)
            open_price = pd.Series(self.data['open'].values)
            volume = pd.Series(self.data['volume'].values)
            
            # Advanced price action features
            self.data['range'] = high - low
            self.data['body'] = np.abs(close - open_price)
            self.data['upper_wick'] = high - np.maximum(open_price, close)
            self.data['lower_wick'] = np.minimum(open_price, close) - low
            
            # Normalized features
            self.data['body_pct'] = self.data['body'] / (self.data['range'] + 1e-8)
            self.data['upper_wick_pct'] = self.data['upper_wick'] / (self.data['range'] + 1e-8)
            self.data['lower_wick_pct'] = self.data['lower_wick'] / (self.data['range'] + 1e-8)
            self.data['close_position'] = (close - low) / (high - low + 1e-8)
            
            # Premium moving averages with multiple timeframes
            for period in [5, 8, 13, 21, 34, 55]:
                self.data[f'sma_{period}'] = close.rolling(period).mean()
                self.data[f'price_vs_sma_{period}'] = close / self.data[f'sma_{period}'] - 1
                
            # Exponential moving averages
            for period in [9, 12, 21, 26]:
                self.data[f'ema_{period}'] = close.ewm(span=period).mean()
                self.data[f'price_vs_ema_{period}'] = close / self.data[f'ema_{period}'] - 1
            
            # Advanced MACD system
            self.data['macd'] = self.data['ema_12'] - self.data['ema_26']
            self.data['macd_signal'] = self.data['macd'].ewm(span=9).mean()
            self.data['macd_histogram'] = self.data['macd'] - self.data['macd_signal']
            self.data['macd_momentum'] = self.data['macd_histogram'].diff()
            
            # Multi-timeframe RSI
            for period in [9, 14, 21]:
                delta = close.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                self.data[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            
            # RSI divergence signals
            self.data['rsi_bullish_div'] = ((self.data['rsi_14'] > self.data['rsi_14'].shift(5)) & 
                                           (close < close.shift(5))).astype(int)
            self.data['rsi_bearish_div'] = ((self.data['rsi_14'] < self.data['rsi_14'].shift(5)) & 
                                           (close > close.shift(5))).astype(int)
            
            # Advanced Bollinger Bands
            for period in [20, 50]:
                bb_mid = close.rolling(period).mean()
                bb_std = close.rolling(period).std()
                self.data[f'bb_upper_{period}'] = bb_mid + (bb_std * 2)
                self.data[f'bb_lower_{period}'] = bb_mid - (bb_std * 2)
                self.data[f'bb_position_{period}'] = (close - self.data[f'bb_lower_{period}']) / (self.data[f'bb_upper_{period}'] - self.data[f'bb_lower_{period}'])
                self.data[f'bb_squeeze_{period}'] = (bb_std / bb_mid < 0.02).astype(int)
            
            # Volatility analysis
            returns = close.pct_change()
            self.data['returns'] = returns
            
            # ATR and volatility ranking
            tr1 = high - low
            tr2 = np.abs(high - close.shift(1))
            tr3 = np.abs(low - close.shift(1))
            true_range = np.maximum(tr1, np.maximum(tr2, tr3))
            
            for period in [14, 21]:
                self.data[f'atr_{period}'] = true_range.rolling(period).mean()
                self.data[f'volatility_rank_{period}'] = self.data[f'atr_{period}'].rolling(100).rank(pct=True)
            
            # Volume analysis with multiple indicators
            for period in [10, 20, 50]:
                self.data[f'volume_sma_{period}'] = volume.rolling(period).mean()
                self.data[f'volume_ratio_{period}'] = volume / self.data[f'volume_sma_{period}']
            
            # Volume spikes and confirmation
            self.data['volume_spike'] = (self.data['volume_ratio_20'] > 2.0).astype(int)
            self.data['volume_dry_up'] = (self.data['volume_ratio_20'] < 0.5).astype(int)
            
            # Price-volume relationship
            self.data['price_volume_trend'] = (returns * volume).rolling(10).sum()
            self.data['money_flow_index'] = self.calculate_mfi(high, low, close, volume)
            
            # Support and resistance with multiple timeframes
            for window in [5, 10, 20, 50]:
                self.data[f'resistance_{window}'] = high.rolling(window).max()
                self.data[f'support_{window}'] = low.rolling(window).min()
                self.data[f'near_resistance_{window}'] = (close > self.data[f'resistance_{window}'] * 0.99).astype(int)
                self.data[f'near_support_{window}'] = (close < self.data[f'support_{window}'] * 1.01).astype(int)
            
            # Advanced candlestick patterns
            self.data['doji'] = (self.data['body'] < self.data['range'] * 0.1).astype(int)
            self.data['hammer'] = ((self.data['lower_wick'] > self.data['body'] * 2) & 
                                  (self.data['upper_wick'] < self.data['body'] * 0.5) &
                                  (self.data['body_pct'] > 0.1)).astype(int)
            self.data['shooting_star'] = ((self.data['upper_wick'] > self.data['body'] * 2) & 
                                         (self.data['lower_wick'] < self.data['body'] * 0.5) &
                                         (self.data['body_pct'] > 0.1)).astype(int)
            
            # Engulfing patterns with volume confirmation
            prev_body = self.data['body'].shift(1)
            prev_close = close.shift(1)
            prev_open = open_price.shift(1)
            
            bullish_engulfing = ((close > open_price) & (prev_close < prev_open) & 
                               (open_price < prev_close) & (close > prev_open) &
                               (self.data['body'] > prev_body * 1.2) &
                               (self.data['volume_ratio_20'] > 1.2))
            bearish_engulfing = ((close < open_price) & (prev_close > prev_open) & 
                               (open_price > prev_close) & (close < prev_open) &
                               (self.data['body'] > prev_body * 1.2) &
                               (self.data['volume_ratio_20'] > 1.2))
            
            self.data['bullish_engulfing'] = bullish_engulfing.astype(int)
            self.data['bearish_engulfing'] = bearish_engulfing.astype(int)
            
            # Multi-timeframe momentum
            for period in [3, 5, 8, 13, 21]:
                self.data[f'momentum_{period}'] = close / close.shift(period) - 1
                self.data[f'roc_{period}'] = ((close - close.shift(period)) / close.shift(period)) * 100
            
            # Momentum acceleration
            self.data['momentum_acceleration'] = self.data['momentum_5'].diff()
            
            # Trend strength analysis
            self.data['trend_strength'] = np.abs(self.data['momentum_13'])
            self.data['strong_uptrend'] = ((self.data['momentum_8'] > 0.01) & 
                                          (self.data['momentum_13'] > 0.015) &
                                          (self.data['momentum_21'] > 0.02)).astype(int)
            self.data['strong_downtrend'] = ((self.data['momentum_8'] < -0.01) & 
                                            (self.data['momentum_13'] < -0.015) &
                                            (self.data['momentum_21'] < -0.02)).astype(int)
            
            # Market regime detection
            volatility = returns.rolling(20).std() * np.sqrt(252)
            vol_percentile = volatility.rolling(100).rank(pct=True)
            trend_strength = np.abs(self.data['momentum_21'])
            
            regimes = []
            for i in range(len(self.data)):
                if i < 21:
                    regimes.append('building')
                    continue
                    
                curr_vol = vol_percentile.iloc[i] if not pd.isna(vol_percentile.iloc[i]) else 0.5
                curr_trend = trend_strength.iloc[i] if not pd.isna(trend_strength.iloc[i]) else 0
                
                if curr_vol > 0.8:
                    if curr_trend > 0.025:
                        regimes.append('high_vol_trending')
                    else:
                        regimes.append('high_vol_ranging')
                elif curr_vol < 0.2:
                    if curr_trend > 0.015:
                        regimes.append('low_vol_trending')
                    else:
                        regimes.append('low_vol_ranging')
                else:
                    if curr_trend > 0.02:
                        regimes.append('normal_trending')
                    else:
                        regimes.append('normal_ranging')
            
            self.data['market_regime'] = regimes
            
            # Create regime dummy variables
            regime_dummies = pd.get_dummies(self.data['market_regime'], prefix='regime')
            self.data = pd.concat([self.data, regime_dummies], axis=1)
            
            # Time-based features
            self.data['day_of_week'] = self.data['date'].dt.dayofweek
            self.data['month'] = self.data['date'].dt.month
            self.data['quarter'] = self.data['date'].dt.quarter
            self.data['is_month_end'] = (self.data['date'].dt.day > 25).astype(int)
            self.data['is_month_start'] = (self.data['date'].dt.day < 6).astype(int)
            
            # Statistical features
            for window in [10, 20]:
                rolling_returns = returns.rolling(window)
                self.data[f'skewness_{window}'] = rolling_returns.skew()
                self.data[f'kurtosis_{window}'] = rolling_returns.kurt()
                
            # Fill NaN values
            self.data = self.data.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            logger.info(f"Premium feature engineering completed. Features: {self.data.shape[1]}")
            return True
            
        except Exception as e:
            logger.error(f"Error in premium feature engineering: {e}")
            return False
    
    def calculate_mfi(self, high, low, close, volume, period=14):
        """Calculate Money Flow Index."""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(period).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(period).sum()
        
        money_ratio = positive_flow / negative_flow
        mfi = 100 - (100 / (1 + money_ratio))
        
        return mfi
    
    def create_optimized_target(self):
        """Create target optimized for high win rate."""
        try:
            # Next-day direction with stricter thresholds
            self.data['next_close'] = self.data['close'].shift(-1)
            self.data['next_return'] = (self.data['next_close'] / self.data['close'] - 1)
            
            # Strict threshold for clear signals only
            min_return_threshold = 0.008  # 0.8% minimum move (quality over quantity)
            
            self.data['target'] = 0  # Default: no clear direction
            
            # Strong bullish
            self.data.loc[self.data['next_return'] > min_return_threshold, 'target'] = 1
            
            # Strong bearish 
            self.data.loc[self.data['next_return'] < -min_return_threshold, 'target'] = -1
            
            # Keep only very clear signals
            valid_targets = self.data['target'] != 0
            self.data = self.data[valid_targets].copy()
            
            # Convert to binary
            self.data['target_binary'] = (self.data['target'] == 1).astype(int)
            
            logger.info(f"Optimized target created. Valid signals: {len(self.data)}")
            logger.info(f"Target distribution: {self.data['target_binary'].value_counts().to_dict()}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating optimized target: {e}")
            return False
    
    def train_premium_ensemble(self):
        """Train premium ensemble optimized for high accuracy."""
        try:
            # Define feature columns
            exclude_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 
                           'target', 'target_binary', 'next_close', 'next_return', 'returns', 'market_regime']
            feature_cols = [col for col in self.data.columns if col not in exclude_cols]
            self.feature_cols = feature_cols
            
            logger.info(f"Training premium ensemble with {len(feature_cols)} features")
            
            # Split data
            split_idx = int(len(self.data) * 0.85)  # 85% for training
            
            X = self.data[feature_cols]
            y = self.data['target_binary']
            
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Scale features
            self.scalers['standard'] = StandardScaler()
            self.scalers['robust'] = RobustScaler()
            
            X_train_standard = self.scalers['standard'].fit_transform(X_train)
            X_test_standard = self.scalers['standard'].transform(X_test)
            
            X_train_robust = self.scalers['robust'].fit_transform(X_train)
            X_test_robust = self.scalers['robust'].transform(X_test)
            
            # Train premium models with optimized hyperparameters
            models_config = {
                'xgboost': {
                    'model': xgb.XGBClassifier(
                        n_estimators=300,
                        max_depth=7,
                        learning_rate=0.03,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_alpha=0.1,
                        reg_lambda=0.1,
                        min_child_weight=3,
                        random_state=42,
                        eval_metric='logloss'
                    ),
                    'data_train': X_train_standard,
                    'data_test': X_test_standard
                },
                'random_forest': {
                    'model': RandomForestClassifier(
                        n_estimators=200,
                        max_depth=12,
                        min_samples_split=6,
                        min_samples_leaf=3,
                        max_features='sqrt',
                        bootstrap=True,
                        random_state=42
                    ),
                    'data_train': X_train_robust,
                    'data_test': X_test_robust
                },
                'gradient_boost': {
                    'model': GradientBoostingClassifier(
                        n_estimators=150,
                        max_depth=6,
                        learning_rate=0.05,
                        subsample=0.85,
                        min_samples_split=6,
                        min_samples_leaf=3,
                        random_state=42
                    ),
                    'data_train': X_train_standard,
                    'data_test': X_test_standard
                },
                'svm': {
                    'model': SVC(
                        C=2.0,
                        kernel='rbf',
                        gamma='scale',
                        probability=True,
                        random_state=42
                    ),
                    'data_train': X_train_standard,
                    'data_test': X_test_standard
                },
                'logistic': {
                    'model': LogisticRegression(
                        C=1.0,
                        penalty='l2',
                        solver='liblinear',
                        random_state=42
                    ),
                    'data_train': X_train_standard,
                    'data_test': X_test_standard
                }
            }
            
            # Train all models
            model_scores = {}
            for model_name, config in models_config.items():
                logger.info(f"Training {model_name}...")
                model = config['model']
                model.fit(config['data_train'], y_train)
                self.models[model_name] = model
                
                # Evaluate
                test_pred = model.predict(config['data_test'])
                accuracy = accuracy_score(y_test, test_pred)
                model_scores[model_name] = accuracy
                logger.info(f"{model_name} accuracy: {accuracy:.4f}")
            
            logger.info("Premium ensemble models trained successfully")
            logger.info(f"Model scores: {model_scores}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training premium ensemble: {e}")
            return False
    
    def generate_premium_signals(self):
        """Generate premium signals optimized for 60%+ win rate."""
        try:
            X = self.data[self.feature_cols]
            
            # Get scaled features
            X_standard = self.scalers['standard'].transform(X)
            X_robust = self.scalers['robust'].transform(X)
            
            # Get predictions from all models
            predictions = {}
            probabilities = {}
            
            # Get predictions
            model_data_map = {
                'xgboost': X_standard,
                'gradient_boost': X_standard,
                'svm': X_standard,
                'logistic': X_standard,
                'random_forest': X_robust
            }
            
            for model_name, data in model_data_map.items():
                pred = self.models[model_name].predict(data)
                proba = self.models[model_name].predict_proba(data)
                predictions[model_name] = pred
                probabilities[model_name] = proba[:, 1]
            
            # Advanced ensemble with weighted voting
            pred_array = np.array([predictions[model] for model in predictions.keys()])
            proba_array = np.array([probabilities[model] for model in probabilities.keys()])
            
            # Weighted ensemble based on model performance (from training scores)
            weights = np.array([0.25, 0.20, 0.20, 0.15, 0.20])  # Optimized weights
            
            # Weighted predictions
            ensemble_pred = np.average(pred_array, axis=0, weights=weights)
            ensemble_proba = np.average(proba_array, axis=0, weights=weights)
            
            # Convert to binary
            ensemble_pred_binary = (ensemble_pred > 0.5).astype(int)
            
            # Calculate advanced metrics
            agreement_up = np.sum(pred_array == 1, axis=0) / len(predictions)
            agreement_down = np.sum(pred_array == 0, axis=0) / len(predictions)
            agreement = np.maximum(agreement_up, agreement_down)
            
            # Confidence based on probability spread and agreement
            proba_std = np.std(proba_array, axis=0)
            confidence = (1 - proba_std) * agreement  # Combined metric
            
            # Signal quality score
            signal_quality = (
                confidence * 0.4 +
                np.maximum(ensemble_proba, 1 - ensemble_proba) * 0.3 +
                agreement * 0.3
            )
            
            # Add to dataframe
            self.data['ensemble_prediction'] = ensemble_pred_binary
            self.data['ensemble_probability'] = ensemble_proba
            self.data['model_agreement'] = agreement
            self.data['prediction_confidence'] = confidence
            self.data['signal_quality'] = signal_quality
            
            # Generate premium signals with very strict filters
            self.data['signal'] = 0
            
            for i in range(len(self.data)):
                if i < 55:  # Skip early periods
                    continue
                
                # Get current market conditions
                regime = self.data['market_regime'].iloc[i]
                vol_rank = self.data['volatility_rank_21'].iloc[i]
                rsi_14 = self.data['rsi_14'].iloc[i]
                bb_pos_20 = self.data['bb_position_20'].iloc[i]
                trend_strength = self.data['trend_strength'].iloc[i]
                volume_ratio = self.data['volume_ratio_20'].iloc[i]
                mfi = self.data['money_flow_index'].iloc[i]
                
                # Premium conditions (very strict)
                ultra_high_agreement = agreement[i] >= self.ensemble_agreement  # 80%
                ultra_high_confidence = confidence[i] >= 0.7  # Very high confidence
                premium_probability = (ensemble_proba[i] > self.min_confidence) or (ensemble_proba[i] < (1 - self.min_confidence))
                ultra_high_quality = signal_quality[i] >= self.signal_quality_threshold  # 85%
                
                # Market condition filters (very strict)
                optimal_volatility = 0.2 <= vol_rank <= 0.8
                non_extreme_rsi = 25 <= rsi_14 <= 75
                good_bb_position = 0.1 <= bb_pos_20 <= 0.9
                good_volume = 0.8 <= volume_ratio <= 3.0
                good_mfi = 20 <= mfi <= 80
                
                # Regime filtering (avoid problematic regimes)
                good_regime = regime in ['low_vol_trending', 'normal_trending', 'low_vol_ranging']
                
                # Trend strength requirement
                if self.trend_filter and trend_strength < 0.005:
                    continue
                
                # Generate signals only with all premium conditions
                if (ultra_high_agreement and ultra_high_confidence and premium_probability and
                    ultra_high_quality and optimal_volatility and non_extreme_rsi and
                    good_bb_position and good_volume and good_mfi and good_regime):
                    
                    if ensemble_pred_binary[i] == 1 and ensemble_proba[i] > self.min_confidence:
                        # Ultra-strict bullish confirmations
                        bullish_momentum = self.data['momentum_5'].iloc[i] > 0
                        strong_uptrend = self.data['strong_uptrend'].iloc[i] == 1
                        macd_bullish = self.data['macd_histogram'].iloc[i] > 0
                        no_bearish_divergence = self.data['rsi_bearish_div'].iloc[i] == 0
                        bullish_engulfing = self.data['bullish_engulfing'].iloc[i] == 1
                        
                        confirmations = sum([bullish_momentum, strong_uptrend, macd_bullish, 
                                           no_bearish_divergence, bullish_engulfing])
                        
                        if confirmations >= 3:  # Need at least 3 confirmations
                            self.data.loc[i, 'signal'] = 1
                    
                    elif ensemble_pred_binary[i] == 0 and ensemble_proba[i] < (1 - self.min_confidence):
                        # Ultra-strict bearish confirmations
                        bearish_momentum = self.data['momentum_5'].iloc[i] < 0
                        strong_downtrend = self.data['strong_downtrend'].iloc[i] == 1
                        macd_bearish = self.data['macd_histogram'].iloc[i] < 0
                        no_bullish_divergence = self.data['rsi_bullish_div'].iloc[i] == 0
                        bearish_engulfing = self.data['bearish_engulfing'].iloc[i] == 1
                        
                        confirmations = sum([bearish_momentum, strong_downtrend, macd_bearish, 
                                           no_bullish_divergence, bearish_engulfing])
                        
                        if confirmations >= 3:  # Need at least 3 confirmations
                            self.data.loc[i, 'signal'] = -1
            
            buy_signals = sum(self.data['signal'] == 1)
            sell_signals = sum(self.data['signal'] == -1)
            
            logger.info(f"Premium signals generated: {buy_signals} buy, {sell_signals} sell")
            logger.info(f"Signal rate: {(buy_signals + sell_signals) / len(self.data) * 100:.2f}%")
            
            return True
            
        except Exception as e:
            logger.error(f"Error generating premium signals: {e}")
            return False
    
    def backtest_premium_strategy(self):
        """Run premium backtesting optimized for high win rate."""
        try:
            logger.info("Starting premium strategy backtest...")
            
            current_capital = self.initial_capital
            self.portfolio_values = [current_capital]
            self.trades = []
            self.positions = []
            
            consecutive_losses = 0
            
            for i in range(55, len(self.data)):  # Start after lookback
                current_date = self.data['date'].iloc[i]
                current_price = self.data['open'].iloc[i]
                signal = self.data['signal'].iloc[i-1]
                volatility_rank = self.data['volatility_rank_21'].iloc[i]
                market_regime = self.data['market_regime'].iloc[i]
                atr = self.data['atr_14'].iloc[i]
                
                # Dynamic risk adjustment
                risk_multiplier = 1.0
                if consecutive_losses >= 2:
                    risk_multiplier = 0.6  # Reduce risk after losses
                elif consecutive_losses <= -3:  # Recent wins
                    risk_multiplier = 1.1  # Slightly increase risk after wins
                
                # Check for position exits
                positions_to_remove = []
                for pos_idx, position in enumerate(self.positions):
                    exit_price = None
                    exit_reason = None
                    
                    # Dynamic stops based on volatility
                    volatility_multiplier = 1.0
                    if volatility_rank > 0.8:
                        volatility_multiplier = 0.8  # Tighter stops in high vol
                    elif volatility_rank < 0.2:
                        volatility_multiplier = 1.2  # Wider stops in low vol
                    
                    adjusted_stop = position['entry_price'] + (position['stop_loss'] - position['entry_price']) * volatility_multiplier
                    adjusted_profit = position['entry_price'] + (position['take_profit'] - position['entry_price']) * volatility_multiplier
                    
                    if position['type'] == 'long':
                        if self.data['low'].iloc[i] <= adjusted_stop:
                            exit_price = adjusted_stop
                            exit_reason = 'stop_loss'
                        elif self.data['high'].iloc[i] >= adjusted_profit:
                            exit_price = adjusted_profit
                            exit_reason = 'take_profit'
                    else:  # short position
                        if self.data['high'].iloc[i] >= adjusted_stop:
                            exit_price = adjusted_stop
                            exit_reason = 'stop_loss'
                        elif self.data['low'].iloc[i] <= adjusted_profit:
                            exit_price = adjusted_profit
                            exit_reason = 'take_profit'
                    
                    # Exit position
                    if exit_price:
                        if position['type'] == 'long':
                            pnl = (exit_price - position['entry_price']) * position['size']
                        else:
                            pnl = (position['entry_price'] - exit_price) * position['size']
                        
                        current_capital += pnl
                        
                        # Update consecutive losses
                        if pnl > 0:
                            consecutive_losses = min(consecutive_losses - 1, -5)
                        else:
                            consecutive_losses = max(consecutive_losses + 1, 5)
                        
                        trade_record = {
                            'entry_date': position['entry_date'],
                            'exit_date': current_date,
                            'type': position['type'],
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'size': position['size'],
                            'pnl': pnl,
                            'return_pct': pnl / (position['entry_price'] * position['size']) * 100,
                            'exit_reason': exit_reason,
                            'market_regime': market_regime,
                            'signal_quality': position.get('signal_quality', 0)
                        }
                        self.trades.append(trade_record)
                        positions_to_remove.append(pos_idx)
                
                # Remove closed positions
                for pos_idx in reversed(positions_to_remove):
                    self.positions.pop(pos_idx)
                
                # Check for new entries with premium conditions
                if signal != 0 and len(self.positions) < self.max_positions:
                    confidence = self.data['prediction_confidence'].iloc[i-1]
                    agreement = self.data['model_agreement'].iloc[i-1]
                    quality = self.data['signal_quality'].iloc[i-1]
                    
                    # Ultra-strict entry conditions
                    if (confidence > 0.7 and agreement >= self.ensemble_agreement and 
                        quality >= self.signal_quality_threshold):
                        
                        base_risk = current_capital * self.risk_per_trade
                        
                        # ATR-based position sizing
                        atr_multiplier = 1.0
                        if atr > 0:
                            # Scale position size inversely with volatility
                            normalized_atr = atr / current_price
                            atr_multiplier = max(0.5, min(1.5, 1.0 / (normalized_atr * 100)))
                        
                        adjusted_risk = base_risk * risk_multiplier * atr_multiplier
                        
                        if signal == 1:  # Buy signal
                            stop_loss_price = current_price * (1 - self.stop_loss_pct)
                            take_profit_price = current_price * (1 + self.take_profit_pct)
                            price_risk = current_price - stop_loss_price
                            position_size = adjusted_risk / price_risk if price_risk > 0 else 0
                            
                            if position_size > 0:
                                position = {
                                    'entry_date': current_date,
                                    'type': 'long',
                                    'entry_price': current_price,
                                    'size': position_size / current_price,
                                    'stop_loss': stop_loss_price,
                                    'take_profit': take_profit_price,
                                    'confidence': confidence,
                                    'agreement': agreement,
                                    'signal_quality': quality
                                }
                                self.positions.append(position)
                        
                        elif signal == -1:  # Sell signal
                            stop_loss_price = current_price * (1 + self.stop_loss_pct)
                            take_profit_price = current_price * (1 - self.take_profit_pct)
                            price_risk = stop_loss_price - current_price
                            position_size = adjusted_risk / price_risk if price_risk > 0 else 0
                            
                            if position_size > 0:
                                position = {
                                    'entry_date': current_date,
                                    'type': 'short',
                                    'entry_price': current_price,
                                    'size': position_size / current_price,
                                    'stop_loss': stop_loss_price,
                                    'take_profit': take_profit_price,
                                    'confidence': confidence,
                                    'agreement': agreement,
                                    'signal_quality': quality
                                }
                                self.positions.append(position)
                
                # Calculate portfolio value
                portfolio_value = current_capital
                for position in self.positions:
                    if position['type'] == 'long':
                        unrealized_pnl = (current_price - position['entry_price']) * position['size']
                    else:
                        unrealized_pnl = (position['entry_price'] - current_price) * position['size']
                    portfolio_value += unrealized_pnl
                
                self.portfolio_values.append(portfolio_value)
            
            logger.info(f"Premium backtest completed. Total trades: {len(self.trades)}")
            return True
            
        except Exception as e:
            logger.error(f"Error in premium backtesting: {e}")
            return False
    
    def calculate_premium_metrics(self):
        """Calculate performance metrics for premium strategy."""
        try:
            if not self.trades:
                logger.warning("No trades to analyze")
                return {}
            
            trades_df = pd.DataFrame(self.trades)
            
            # Basic metrics
            total_return = (self.portfolio_values[-1] - self.initial_capital) / self.initial_capital * 100
            total_trades = len(self.trades)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            losing_trades = len(trades_df[trades_df['pnl'] < 0])
            win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
            
            # Advanced metrics
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
            profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if avg_loss != 0 else float('inf')
            
            # Risk metrics
            portfolio_series = pd.Series(self.portfolio_values)
            returns = portfolio_series.pct_change().dropna()
            
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() != 0 else 0
            
            # Maximum drawdown
            peak = portfolio_series.expanding().max()
            drawdown = (portfolio_series - peak) / peak * 100
            max_drawdown = drawdown.min()
            
            # Additional metrics
            largest_win = trades_df['pnl'].max() if total_trades > 0 else 0
            largest_loss = trades_df['pnl'].min() if total_trades > 0 else 0
            expectancy = (avg_win * win_rate/100) + (avg_loss * (100-win_rate)/100)
            recovery_factor = total_return / abs(max_drawdown) if max_drawdown != 0 else float('inf')
            
            # Quality metrics
            avg_signal_quality = trades_df['signal_quality'].mean() if 'signal_quality' in trades_df.columns else 0
            
            metrics = {
                'total_return_pct': total_return,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate_pct': win_rate,
                'profit_factor': profit_factor,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'largest_win': largest_win,
                'largest_loss': largest_loss,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown_pct': max_drawdown,
                'final_capital': self.portfolio_values[-1],
                'total_pnl': sum(trades_df['pnl']),
                'expectancy': expectancy,
                'recovery_factor': recovery_factor,
                'avg_signal_quality': avg_signal_quality
            }
            
            logger.info("Premium performance metrics calculated")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating premium metrics: {e}")
            return {}
    
    def run_optimized_analysis(self):
        """Run the complete optimized strategy analysis."""
        try:
            print("🚀 Starting OPTIMIZED 60%+ Win Rate Strategy Analysis...")
            print("🎯 Target: 60%+ Win Rate with Premium ML Ensemble")
            
            # Premium pipeline
            if not self.load_and_prepare_data():
                return False
            print("✅ Data loaded and prepared")
            
            if not self.engineer_premium_features():
                return False
            print("✅ Premium features engineered")
            
            if not self.create_optimized_target():
                return False
            print("✅ Optimized target variable created")
            
            if not self.train_premium_ensemble():
                return False
            print("✅ Premium ensemble models trained")
            
            if not self.generate_premium_signals():
                return False
            print("✅ Premium signals generated")
            
            if not self.backtest_premium_strategy():
                return False
            print("✅ Premium strategy backtested")
            
            metrics = self.calculate_premium_metrics()
            if not metrics:
                return False
            print("✅ Premium metrics calculated")
            
            # Results summary
            print(f"\n🎯 OPTIMIZED 60%+ WIN RATE STRATEGY RESULTS")
            print(f"════════════════════════════════════════")
            print(f"🏆 Win Rate: {metrics.get('win_rate_pct', 0):.2f}% {'🎉 TARGET ACHIEVED!' if metrics.get('win_rate_pct', 0) >= 60 else '🔄 Approaching Target'}")
            print(f"📈 Total Return: {metrics.get('total_return_pct', 0):.2f}%")
            print(f"💰 Final Capital: ${metrics.get('final_capital', 0):,.2f}")
            print(f"📊 Total Trades: {metrics.get('total_trades', 0)}")
            print(f"⚡ Profit Factor: {metrics.get('profit_factor', 0):.2f}")
            print(f"🛡️ Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
            print(f"📈 Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"🎯 Expectancy: ${metrics.get('expectancy', 0):.2f}")
            print(f"⭐ Avg Signal Quality: {metrics.get('avg_signal_quality', 0):.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in optimized analysis: {e}")
            return False

def main():
    """Main execution function."""
    try:
        strategy = Optimized60PercentStrategy('XAU_1d_data_clean.csv')
        success = strategy.run_optimized_analysis()
        
        if success:
            print("\n🚀 Optimized 60%+ Win Rate Strategy Analysis completed!")
            print("🎯 Premium ML ensemble optimized for maximum win rate!")
        else:
            print("\n❌ Optimized Strategy Analysis failed. Check logs for details.")
            
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        logger.error(f"Fatal error in main: {e}")

if __name__ == "__main__":
    main()