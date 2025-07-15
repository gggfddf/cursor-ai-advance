#!/usr/bin/env python3
"""
Enhanced Lorentzian Classification ML Backtesting System
Optimized version with improved parameters and additional features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class EnhancedTechnicalIndicators:
    """Enhanced technical indicators with additional features"""
    
    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """MACD Histogram"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        return macd_line - signal_line
    
    @staticmethod
    def bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> pd.Series:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=d_period).mean()
        return d_percent
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Williams %R"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr
    
    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """Commodity Channel Index"""
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        cci = (tp - sma_tp) / (0.015 * mad)
        return cci
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average Directional Index"""
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = minus_dm.abs()
        
        tr_list = []
        for i in range(len(close)):
            if i == 0:
                tr_list.append(high.iloc[i] - low.iloc[i])
            else:
                tr_list.append(max(
                    high.iloc[i] - low.iloc[i],
                    abs(high.iloc[i] - close.iloc[i-1]),
                    abs(low.iloc[i] - close.iloc[i-1])
                ))
        
        tr = pd.Series(tr_list, index=close.index)
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        return dx.rolling(window=period).mean()
    
    @staticmethod
    def wt(hlc3: pd.Series, period1: int = 10, period2: int = 21) -> pd.Series:
        """Wave Trend"""
        esa = hlc3.ewm(span=period1).mean()
        d = (hlc3 - esa).abs().ewm(span=period1).mean()
        ci = (hlc3 - esa) / (0.015 * d)
        return ci.ewm(span=period2).mean()
    
    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Volume Weighted Average Price"""
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()
    
    @staticmethod
    def momentum(prices: pd.Series, period: int = 10) -> pd.Series:
        """Price Momentum"""
        return prices.diff(period) / prices.shift(period) * 100

class OptimizedLorentzianClassifier:
    """Optimized Lorentzian Classification ML System"""
    
    def __init__(self, 
                 neighbors_count: int = 12,
                 max_bars_back: int = 3000,
                 feature_count: int = 8,
                 prediction_threshold: float = 0.5,
                 use_volatility_filter: bool = True,
                 use_regime_filter: bool = True,
                 use_adx_filter: bool = True,
                 regime_threshold: float = 0.1,
                 adx_threshold: int = 25,
                 use_ema_filter: bool = True,
                 ema_period: int = 50,
                 use_sma_filter: bool = True,
                 sma_period: int = 200):
        
        self.neighbors_count = neighbors_count
        self.max_bars_back = max_bars_back
        self.feature_count = feature_count
        self.prediction_threshold = prediction_threshold
        self.use_volatility_filter = use_volatility_filter
        self.use_regime_filter = use_regime_filter
        self.use_adx_filter = use_adx_filter
        self.regime_threshold = regime_threshold
        self.adx_threshold = adx_threshold
        self.use_ema_filter = use_ema_filter
        self.ema_period = ema_period
        self.use_sma_filter = use_sma_filter
        self.sma_period = sma_period
        
        # Enhanced feature settings
        self.features = {
            'f1': {'name': 'RSI', 'param_a': 14, 'param_b': 1},
            'f2': {'name': 'MACD', 'param_a': 12, 'param_b': 26},
            'f3': {'name': 'CCI', 'param_a': 20, 'param_b': 1},
            'f4': {'name': 'ADX', 'param_a': 14, 'param_b': 1},
            'f5': {'name': 'STOCH', 'param_a': 14, 'param_b': 3},
            'f6': {'name': 'WT', 'param_a': 10, 'param_b': 21},
            'f7': {'name': 'WILLIAMS', 'param_a': 14, 'param_b': 1},
            'f8': {'name': 'MOMENTUM', 'param_a': 10, 'param_b': 1}
        }
        
        self.indicators = EnhancedTechnicalIndicators()
        
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all enhanced features"""
        data = df.copy()
        data['hlc3'] = (data['high'] + data['low'] + data['close']) / 3
        
        # Calculate features
        for i in range(1, self.feature_count + 1):
            feature_config = self.features[f'f{i}']
            feature_name = feature_config['name']
            param_a = feature_config['param_a']
            param_b = feature_config['param_b']
            
            if feature_name == 'RSI':
                data[f'f{i}'] = self.indicators.rsi(data['close'], param_a)
            elif feature_name == 'MACD':
                data[f'f{i}'] = self.indicators.macd(data['close'], param_a, param_b)
            elif feature_name == 'CCI':
                data[f'f{i}'] = self.indicators.cci(data['high'], data['low'], data['close'], param_a)
            elif feature_name == 'ADX':
                data[f'f{i}'] = self.indicators.adx(data['high'], data['low'], data['close'], param_a)
            elif feature_name == 'STOCH':
                data[f'f{i}'] = self.indicators.stochastic(data['high'], data['low'], data['close'], param_a, param_b)
            elif feature_name == 'WT':
                data[f'f{i}'] = self.indicators.wt(data['hlc3'], param_a, param_b)
            elif feature_name == 'WILLIAMS':
                data[f'f{i}'] = self.indicators.williams_r(data['high'], data['low'], data['close'], param_a)
            elif feature_name == 'MOMENTUM':
                data[f'f{i}'] = self.indicators.momentum(data['close'], param_a)
        
        # Normalize features to [-1, 1] range
        for i in range(1, self.feature_count + 1):
            feature_col = f'f{i}'
            if feature_col in data.columns:
                data[feature_col] = self.normalize_feature(data[feature_col])
        
        return data
    
    def normalize_feature(self, feature: pd.Series) -> pd.Series:
        """Normalize feature to [-1, 1] range"""
        feature_clean = feature.dropna()
        if len(feature_clean) == 0:
            return feature
        
        feature_min = feature_clean.min()
        feature_max = feature_clean.max()
        
        if feature_max == feature_min:
            return pd.Series(0, index=feature.index)
        
        normalized = 2 * (feature - feature_min) / (feature_max - feature_min) - 1
        return normalized
    
    def get_lorentzian_distance(self, current_features: List[float], historical_features: List[float]) -> float:
        """Enhanced Lorentzian distance calculation"""
        distance = 0.0
        weights = [1.2, 1.0, 0.8, 1.1, 0.9, 1.0, 0.7, 0.8]  # Feature importance weights
        
        for i in range(len(current_features)):
            weight = weights[i] if i < len(weights) else 1.0
            distance += weight * np.log(1 + abs(current_features[i] - historical_features[i]))
        
        return distance
    
    def enhanced_volatility_filter(self, df: pd.DataFrame) -> pd.Series:
        """Enhanced volatility filter"""
        if not self.use_volatility_filter:
            return pd.Series(True, index=df.index)
        
        # Calculate multiple volatility measures
        atr = self.calculate_atr(df['high'], df['low'], df['close'])
        bb_upper, bb_middle, bb_lower = self.indicators.bollinger_bands(df['close'])
        bb_width = (bb_upper - bb_lower) / bb_middle
        
        # Combined volatility condition
        atr_condition = atr < atr.rolling(window=20).mean() * 1.2
        bb_condition = bb_width < bb_width.rolling(window=20).mean() * 1.5
        
        return atr_condition & bb_condition
    
    def enhanced_regime_filter(self, df: pd.DataFrame) -> pd.Series:
        """Enhanced regime filter"""
        if not self.use_regime_filter:
            return pd.Series(True, index=df.index)
        
        # Multiple timeframe trend analysis
        sma_fast = df['close'].rolling(window=10).mean()
        sma_slow = df['close'].rolling(window=30).mean()
        ema_fast = df['close'].ewm(span=12).mean()
        ema_slow = df['close'].ewm(span=26).mean()
        
        trend_strength = (sma_fast - sma_slow) / df['close']
        ema_trend = (ema_fast - ema_slow) / df['close']
        
        return (trend_strength > self.regime_threshold) & (ema_trend > self.regime_threshold * 0.5)
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range"""
        tr_list = []
        for i in range(len(close)):
            if i == 0:
                tr_list.append(high.iloc[i] - low.iloc[i])
            else:
                tr_list.append(max(
                    high.iloc[i] - low.iloc[i],
                    abs(high.iloc[i] - close.iloc[i-1]),
                    abs(low.iloc[i] - close.iloc[i-1])
                ))
        
        tr = pd.Series(tr_list, index=close.index)
        return tr.rolling(window=period).mean()
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate enhanced ML-based trading signals"""
        data = self.calculate_features(df)
        
        # Calculate training labels (3 bars ahead for faster signals)
        data['y_train'] = np.where(
            data['close'].shift(-3) > data['close'] * 1.001, 1,  # 0.1% threshold
            np.where(data['close'].shift(-3) < data['close'] * 0.999, -1, 0)
        )
        
        predictions = []
        signals = []
        
        # Enhanced filters
        vol_filter = self.enhanced_volatility_filter(data)
        regime_filter = self.enhanced_regime_filter(data)
        adx_filter = self.indicators.adx(data['high'], data['low'], data['close']) > self.adx_threshold if self.use_adx_filter else pd.Series(True, index=data.index)
        
        # EMA/SMA filters
        ema = data['close'].ewm(span=self.ema_period).mean() if self.use_ema_filter else data['close']
        sma = data['close'].rolling(window=self.sma_period).mean() if self.use_sma_filter else data['close']
        
        ema_up = data['close'] > ema if self.use_ema_filter else pd.Series(True, index=data.index)
        ema_down = data['close'] < ema if self.use_ema_filter else pd.Series(True, index=data.index)
        sma_up = data['close'] > sma if self.use_sma_filter else pd.Series(True, index=data.index)
        sma_down = data['close'] < sma if self.use_sma_filter else pd.Series(True, index=data.index)
        
        # Combined filters
        long_filter = vol_filter & regime_filter & adx_filter & ema_up & sma_up
        short_filter = vol_filter & regime_filter & adx_filter & ema_down & sma_down
        
        # Enhanced ML Logic
        for i in range(len(data)):
            if i < max(100, self.max_bars_back // 10):  # Minimum warmup period
                predictions.append(0)
                signals.append(0)
                continue
            
            # Get current features
            current_features = []
            for j in range(1, self.feature_count + 1):
                if f'f{j}' in data.columns:
                    current_features.append(data[f'f{j}'].iloc[i])
            
            if len(current_features) == 0 or any(pd.isna(current_features)):
                predictions.append(0)
                signals.append(0)
                continue
            
            # Find nearest neighbors with enhanced sampling
            distances = []
            neighbor_labels = []
            
            start_idx = max(0, i - self.max_bars_back)
            
            # Use adaptive sampling based on recent market conditions
            for j in range(start_idx, i, 2):  # Every 2 bars for better coverage
                if j < 0 or j >= len(data):
                    continue
                
                historical_features = []
                for k in range(1, self.feature_count + 1):
                    if f'f{k}' in data.columns:
                        historical_features.append(data[f'f{k}'].iloc[j])
                
                if len(historical_features) == 0 or any(pd.isna(historical_features)):
                    continue
                
                distance = self.get_lorentzian_distance(current_features, historical_features)
                distances.append(distance)
                neighbor_labels.append(data['y_train'].iloc[j])
            
            # Enhanced prediction with confidence scoring
            if len(distances) > 0:
                sorted_pairs = sorted(zip(distances, neighbor_labels))
                k_nearest = sorted_pairs[:self.neighbors_count]
                
                # Weighted prediction based on distance
                total_weight = 0
                weighted_prediction = 0
                
                for dist, label in k_nearest:
                    weight = 1 / (1 + dist)  # Closer neighbors get higher weight
                    weighted_prediction += weight * label
                    total_weight += weight
                
                if total_weight > 0:
                    prediction = weighted_prediction / total_weight
                else:
                    prediction = 0
                
                predictions.append(prediction)
                
                # Enhanced signal generation with confidence threshold
                if prediction > self.prediction_threshold and long_filter.iloc[i]:
                    signal = 1
                elif prediction < -self.prediction_threshold and short_filter.iloc[i]:
                    signal = -1
                else:
                    signal = 0
                
                signals.append(signal)
            else:
                predictions.append(0)
                signals.append(0)
        
        data['prediction'] = predictions
        data['signal'] = signals
        
        # Generate entry/exit signals with improved logic
        data['signal_change'] = data['signal'].diff()
        data['long_entry'] = (data['signal'] == 1) & (data['signal_change'] != 0)
        data['short_entry'] = (data['signal'] == -1) & (data['signal_change'] != 0)
        
        # Enhanced exit logic
        data['long_exit'] = False
        data['short_exit'] = False
        
        position = 0
        bars_held = 0
        
        for i in range(len(data)):
            if data['long_entry'].iloc[i] and position == 0:
                position = 1
                bars_held = 0
            elif data['short_entry'].iloc[i] and position == 0:
                position = -1
                bars_held = 0
            elif position != 0:
                bars_held += 1
                
                # Exit conditions: signal change, max hold period, or stop conditions
                exit_signal_change = data['signal_change'].iloc[i] != 0
                exit_max_hold = bars_held >= 5  # Max 5 bars hold
                
                if exit_signal_change or exit_max_hold:
                    if position == 1:
                        data.loc[data.index[i], 'long_exit'] = True
                    else:
                        data.loc[data.index[i], 'short_exit'] = True
                    position = 0
                    bars_held = 0
        
        return data

class OptimizedBacktestEngine:
    """Enhanced backtesting engine with improved risk management"""
    
    def __init__(self, initial_capital: float = 10000, commission: float = 0.001, 
                 stop_loss_pct: float = 0.02, take_profit_pct: float = 0.04):
        self.initial_capital = initial_capital
        self.commission = commission
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
    def run_backtest(self, df: pd.DataFrame) -> Dict:
        """Run enhanced backtest with risk management"""
        results = {
            'trades': [],
            'equity_curve': [],
            'statistics': {}
        }
        
        capital = self.initial_capital
        position = 0
        entry_price = 0
        entry_date = None
        stop_loss = 0
        take_profit = 0
        
        for i in range(len(df)):
            current_price = df['close'].iloc[i]
            current_date = df.index[i]
            
            # Check stop loss and take profit
            if position == 1:  # Long position
                if current_price <= stop_loss or current_price >= take_profit:
                    # Force exit
                    pnl = current_price - entry_price
                    pnl_pct = (pnl / entry_price) * 100
                    commission_cost = entry_price * self.commission * 2
                    net_pnl = pnl - commission_cost
                    capital += net_pnl
                    
                    exit_reason = "Stop Loss" if current_price <= stop_loss else "Take Profit"
                    
                    results['trades'].append({
                        'entry_date': entry_date,
                        'exit_date': current_date,
                        'type': 'long',
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'net_pnl': net_pnl,
                        'capital': capital,
                        'exit_reason': exit_reason
                    })
                    
                    position = 0
                    continue
            
            elif position == -1:  # Short position
                if current_price >= stop_loss or current_price <= take_profit:
                    # Force exit
                    pnl = entry_price - current_price
                    pnl_pct = (pnl / entry_price) * 100
                    commission_cost = entry_price * self.commission * 2
                    net_pnl = pnl - commission_cost
                    capital += net_pnl
                    
                    exit_reason = "Stop Loss" if current_price >= stop_loss else "Take Profit"
                    
                    results['trades'].append({
                        'entry_date': entry_date,
                        'exit_date': current_date,
                        'type': 'short',
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'net_pnl': net_pnl,
                        'capital': capital,
                        'exit_reason': exit_reason
                    })
                    
                    position = 0
                    continue
            
            # Regular entry/exit logic
            if df['long_entry'].iloc[i] and position == 0:
                position = 1
                entry_price = current_price
                entry_date = current_date
                stop_loss = entry_price * (1 - self.stop_loss_pct)
                take_profit = entry_price * (1 + self.take_profit_pct)
                
            elif df['short_entry'].iloc[i] and position == 0:
                position = -1
                entry_price = current_price
                entry_date = current_date
                stop_loss = entry_price * (1 + self.stop_loss_pct)
                take_profit = entry_price * (1 - self.take_profit_pct)
                
            elif df['long_exit'].iloc[i] and position == 1:
                pnl = current_price - entry_price
                pnl_pct = (pnl / entry_price) * 100
                commission_cost = entry_price * self.commission * 2
                net_pnl = pnl - commission_cost
                capital += net_pnl
                
                results['trades'].append({
                    'entry_date': entry_date,
                    'exit_date': current_date,
                    'type': 'long',
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'net_pnl': net_pnl,
                    'capital': capital,
                    'exit_reason': 'Signal Exit'
                })
                
                position = 0
                
            elif df['short_exit'].iloc[i] and position == -1:
                pnl = entry_price - current_price
                pnl_pct = (pnl / entry_price) * 100
                commission_cost = entry_price * self.commission * 2
                net_pnl = pnl - commission_cost
                capital += net_pnl
                
                results['trades'].append({
                    'entry_date': entry_date,
                    'exit_date': current_date,
                    'type': 'short',
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'net_pnl': net_pnl,
                    'capital': capital,
                    'exit_reason': 'Signal Exit'
                })
                
                position = 0
            
            results['equity_curve'].append({
                'date': current_date,
                'capital': capital,
                'position': position
            })
        
        results['statistics'] = self.calculate_statistics(results['trades'])
        return results
    
    def calculate_statistics(self, trades: List[Dict]) -> Dict:
        """Calculate comprehensive statistics"""
        if not trades:
            return {}
        
        df_trades = pd.DataFrame(trades)
        
        total_trades = len(trades)
        winning_trades = df_trades[df_trades['net_pnl'] > 0]
        losing_trades = df_trades[df_trades['net_pnl'] <= 0]
        
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        total_pnl = df_trades['net_pnl'].sum()
        total_return = (df_trades['capital'].iloc[-1] - self.initial_capital) / self.initial_capital * 100
        
        avg_win = winning_trades['net_pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['net_pnl'].mean() if len(losing_trades) > 0 else 0
        
        total_wins = winning_trades['net_pnl'].sum() if len(winning_trades) > 0 else 0
        total_losses = abs(losing_trades['net_pnl'].sum()) if len(losing_trades) > 0 else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Enhanced statistics
        equity_curve = df_trades['capital'].values
        returns = df_trades['net_pnl'] / self.initial_capital
        
        # Maximum drawdown
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        # Sharpe ratio
        if returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Calmar ratio
        calmar_ratio = (total_return / abs(max_drawdown)) if max_drawdown != 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'final_capital': df_trades['capital'].iloc[-1] if len(df_trades) > 0 else self.initial_capital
        }

def main():
    """Main execution function"""
    print("🚀 Loading XAUUSD Data...")
    try:
        df = pd.read_csv('XAU_1d_data_clean.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.columns = [col.lower() for col in df.columns]
        print(f"✅ Data loaded: {len(df)} rows")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Test multiple configurations
    configs = [
        {
            'name': 'Balanced',
            'neighbors': 10,
            'features': 8,
            'threshold': 0.3,
            'stop_loss': 0.015,
            'take_profit': 0.035
        },
        {
            'name': 'Conservative',
            'neighbors': 15,
            'features': 6,
            'threshold': 0.5,
            'stop_loss': 0.01,
            'take_profit': 0.025
        },
        {
            'name': 'Aggressive',
            'neighbors': 8,
            'features': 8,
            'threshold': 0.2,
            'stop_loss': 0.02,
            'take_profit': 0.05
        }
    ]
    
    best_config = None
    best_sharpe = -999
    
    for config in configs:
        print(f"\n🤖 Testing {config['name']} Configuration...")
        
        classifier = OptimizedLorentzianClassifier(
            neighbors_count=config['neighbors'],
            feature_count=config['features'],
            prediction_threshold=config['threshold'],
            use_volatility_filter=True,
            use_regime_filter=True,
            use_adx_filter=True,
            use_ema_filter=True,
            use_sma_filter=True
        )
        
        df_signals = classifier.generate_signals(df)
        
        backtest_engine = OptimizedBacktestEngine(
            initial_capital=10000,
            commission=0.001,
            stop_loss_pct=config['stop_loss'],
            take_profit_pct=config['take_profit']
        )
        
        results = backtest_engine.run_backtest(df_signals)
        stats = results['statistics']
        
        print(f"📊 {config['name']} Results:")
        print(f"   Total Trades: {stats.get('total_trades', 0)}")
        print(f"   Win Rate: {stats.get('win_rate', 0):.2f}%")
        print(f"   Total Return: {stats.get('total_return', 0):.2f}%")
        print(f"   Sharpe Ratio: {stats.get('sharpe_ratio', 0):.2f}")
        print(f"   Profit Factor: {stats.get('profit_factor', 0):.2f}")
        print(f"   Max Drawdown: {stats.get('max_drawdown', 0):.2f}%")
        
        # Track best configuration
        if stats.get('sharpe_ratio', -999) > best_sharpe:
            best_sharpe = stats.get('sharpe_ratio', -999)
            best_config = config
            best_results = results
            best_signals = df_signals
    
    # Display best results
    print("\n" + "="*70)
    print(f"🏆 BEST CONFIGURATION: {best_config['name']}")
    print("="*70)
    
    stats = best_results['statistics']
    print(f"📊 Total Trades: {stats.get('total_trades', 0)}")
    print(f"🏆 Win Rate: {stats.get('win_rate', 0):.2f}%")
    print(f"💰 Total Return: {stats.get('total_return', 0):.2f}%")
    print(f"💵 Final Capital: ${stats.get('final_capital', 0):,.2f}")
    print(f"⚡ Profit Factor: {stats.get('profit_factor', 0):.2f}")
    print(f"📊 Sharpe Ratio: {stats.get('sharpe_ratio', 0):.2f}")
    print(f"🎯 Calmar Ratio: {stats.get('calmar_ratio', 0):.2f}")
    print(f"📉 Max Drawdown: {stats.get('max_drawdown', 0):.2f}%")
    print(f"🎪 Average Win: ${stats.get('avg_win', 0):.2f}")
    print(f"⚠️ Average Loss: ${stats.get('avg_loss', 0):.2f}")
    
    # Export best results
    if best_results['trades']:
        trades_df = pd.DataFrame(best_results['trades'])
        filename = f"lorentzian_optimized_{best_config['name'].lower()}_trades.csv"
        trades_df.to_csv(filename, index=False)
        print(f"\n📁 Best trades exported to: {filename}")

if __name__ == "__main__":
    main()