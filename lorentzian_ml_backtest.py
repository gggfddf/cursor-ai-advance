#!/usr/bin/env python3
"""
Lorentzian Classification ML Backtesting System
Based on Pine Script: Machine Learning: Lorentzian Classification by @jdehorty

This Python implementation replicates the Pine Script logic for backtesting purposes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class TechnicalIndicators:
    """Technical indicators calculation class"""
    
    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
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
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    @staticmethod
    def wt(hlc3: pd.Series, period1: int = 10, period2: int = 21) -> pd.Series:
        """Wave Trend (approximation)"""
        esa = hlc3.ewm(span=period1).mean()
        d = (hlc3 - esa).abs().ewm(span=period1).mean()
        ci = (hlc3 - esa) / (0.015 * d)
        wt1 = ci.ewm(span=period2).mean()
        return wt1

class LorentzianClassifier:
    """Main Lorentzian Classification ML System"""
    
    def __init__(self, 
                 neighbors_count: int = 8,
                 max_bars_back: int = 2000,
                 feature_count: int = 5,
                 use_volatility_filter: bool = True,
                 use_regime_filter: bool = True,
                 use_adx_filter: bool = False,
                 regime_threshold: float = -0.1,
                 adx_threshold: int = 20,
                 use_ema_filter: bool = False,
                 ema_period: int = 200,
                 use_sma_filter: bool = False,
                 sma_period: int = 200):
        
        self.neighbors_count = neighbors_count
        self.max_bars_back = max_bars_back
        self.feature_count = feature_count
        self.use_volatility_filter = use_volatility_filter
        self.use_regime_filter = use_regime_filter
        self.use_adx_filter = use_adx_filter
        self.regime_threshold = regime_threshold
        self.adx_threshold = adx_threshold
        self.use_ema_filter = use_ema_filter
        self.ema_period = ema_period
        self.use_sma_filter = use_sma_filter
        self.sma_period = sma_period
        
        # Feature settings (mimicking Pine Script defaults)
        self.features = {
            'f1': {'name': 'RSI', 'param_a': 14, 'param_b': 1},
            'f2': {'name': 'WT', 'param_a': 10, 'param_b': 11},
            'f3': {'name': 'CCI', 'param_a': 20, 'param_b': 1},
            'f4': {'name': 'ADX', 'param_a': 20, 'param_b': 2},
            'f5': {'name': 'RSI', 'param_a': 9, 'param_b': 1}
        }
        
        self.indicators = TechnicalIndicators()
        
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all features used in the ML model"""
        data = df.copy()
        
        # Calculate HLC3
        data['hlc3'] = (data['high'] + data['low'] + data['close']) / 3
        
        # Calculate features
        for i in range(1, self.feature_count + 1):
            feature_config = self.features[f'f{i}']
            feature_name = feature_config['name']
            param_a = feature_config['param_a']
            param_b = feature_config['param_b']
            
            if feature_name == 'RSI':
                data[f'f{i}'] = self.indicators.rsi(data['close'], param_a)
            elif feature_name == 'WT':
                data[f'f{i}'] = self.indicators.wt(data['hlc3'], param_a, param_b)
            elif feature_name == 'CCI':
                data[f'f{i}'] = self.indicators.cci(data['high'], data['low'], data['close'], param_a)
            elif feature_name == 'ADX':
                data[f'f{i}'] = self.indicators.adx(data['high'], data['low'], data['close'], param_a)
        
        return data
    
    def get_lorentzian_distance(self, current_features: List[float], historical_features: List[float]) -> float:
        """Calculate Lorentzian distance between current and historical features"""
        distance = 0.0
        for i in range(len(current_features)):
            distance += np.log(1 + abs(current_features[i] - historical_features[i]))
        return distance
    
    def volatility_filter(self, df: pd.DataFrame) -> pd.Series:
        """Volatility filter logic"""
        if not self.use_volatility_filter:
            return pd.Series(True, index=df.index)
        
        # Simple volatility filter using ATR
        atr = self.calculate_atr(df['high'], df['low'], df['close'])
        volatility_threshold = atr.rolling(window=10).mean()
        return atr < volatility_threshold * 1.5
    
    def regime_filter(self, df: pd.DataFrame) -> pd.Series:
        """Regime filter logic"""
        if not self.use_regime_filter:
            return pd.Series(True, index=df.index)
        
        # Simplified regime filter
        sma_fast = df['close'].rolling(window=8).mean()
        sma_slow = df['close'].rolling(window=24).mean()
        regime_value = (sma_fast - sma_slow) / df['close']
        return regime_value > self.regime_threshold
    
    def adx_filter(self, df: pd.DataFrame) -> pd.Series:
        """ADX filter logic"""
        if not self.use_adx_filter:
            return pd.Series(True, index=df.index)
        
        adx = self.indicators.adx(df['high'], df['low'], df['close'])
        return adx > self.adx_threshold
    
    def ema_filter(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """EMA filter logic"""
        if not self.use_ema_filter:
            return pd.Series(True, index=df.index), pd.Series(True, index=df.index)
        
        ema = df['close'].ewm(span=self.ema_period).mean()
        uptrend = df['close'] > ema
        downtrend = df['close'] < ema
        return uptrend, downtrend
    
    def sma_filter(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """SMA filter logic"""
        if not self.use_sma_filter:
            return pd.Series(True, index=df.index), pd.Series(True, index=df.index)
        
        sma = df['close'].rolling(window=self.sma_period).mean()
        uptrend = df['close'] > sma
        downtrend = df['close'] < sma
        return uptrend, downtrend
    
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
        """Generate ML-based trading signals"""
        data = self.calculate_features(df)
        
        # Calculate training labels (4 bars ahead prediction)
        data['y_train'] = np.where(
            data['close'].shift(-4) > data['close'], 1,
            np.where(data['close'].shift(-4) < data['close'], -1, 0)
        )
        
        # Initialize arrays for ML logic
        predictions = []
        signals = []
        
        # Calculate filters
        vol_filter = self.volatility_filter(data)
        regime_filter = self.regime_filter(data)
        adx_filter = self.adx_filter(data)
        ema_up, ema_down = self.ema_filter(data)
        sma_up, sma_down = self.sma_filter(data)
        
        # Combined filter
        filter_all = vol_filter & regime_filter & adx_filter
        
        # ML Logic - Approximate Nearest Neighbors with Lorentzian Distance
        for i in range(len(data)):
            if i < self.max_bars_back:
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
            
            # Find nearest neighbors
            distances = []
            neighbor_labels = []
            
            # Look back through historical data
            start_idx = max(0, i - self.max_bars_back)
            for j in range(start_idx, i, 4):  # Every 4 bars for chronological spacing
                if j < 0 or j >= len(data):
                    continue
                
                # Get historical features
                historical_features = []
                for k in range(1, self.feature_count + 1):
                    if f'f{k}' in data.columns:
                        historical_features.append(data[f'f{k}'].iloc[j])
                
                if len(historical_features) == 0 or any(pd.isna(historical_features)):
                    continue
                
                # Calculate Lorentzian distance
                distance = self.get_lorentzian_distance(current_features, historical_features)
                
                # Store distance and corresponding label
                distances.append(distance)
                neighbor_labels.append(data['y_train'].iloc[j])
            
            # Get k nearest neighbors
            if len(distances) > 0:
                # Sort by distance and take k nearest
                sorted_pairs = sorted(zip(distances, neighbor_labels))
                k_nearest = sorted_pairs[:self.neighbors_count]
                
                # Prediction is sum of k nearest neighbor labels
                prediction = sum([label for _, label in k_nearest])
                predictions.append(prediction)
                
                # Generate signal based on prediction and filters
                if prediction > 0 and filter_all.iloc[i]:
                    signal = 1  # Long
                elif prediction < 0 and filter_all.iloc[i]:
                    signal = -1  # Short
                else:
                    signal = signals[-1] if signals else 0  # Hold previous signal
                
                signals.append(signal)
            else:
                predictions.append(0)
                signals.append(0)
        
        data['prediction'] = predictions
        data['signal'] = signals
        
        # Generate entry/exit signals
        data['signal_change'] = data['signal'].diff()
        data['long_entry'] = (data['signal'] == 1) & (data['signal_change'] != 0) & ema_up & sma_up
        data['short_entry'] = (data['signal'] == -1) & (data['signal_change'] != 0) & ema_down & sma_down
        
        # Exit signals (simplified - after 4 bars or signal change)
        data['long_exit'] = False
        data['short_exit'] = False
        
        # Track positions and exits
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
                
                # Exit after 4 bars or signal change
                if bars_held >= 4 or data['signal_change'].iloc[i] != 0:
                    if position == 1:
                        data.loc[data.index[i], 'long_exit'] = True
                    else:
                        data.loc[data.index[i], 'short_exit'] = True
                    position = 0
                    bars_held = 0
        
        return data

class BacktestEngine:
    """Backtesting engine for the Lorentzian Classification strategy"""
    
    def __init__(self, initial_capital: float = 10000, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        
    def run_backtest(self, df: pd.DataFrame) -> Dict:
        """Run complete backtest"""
        results = {
            'trades': [],
            'equity_curve': [],
            'statistics': {}
        }
        
        capital = self.initial_capital
        position = 0
        entry_price = 0
        entry_date = None
        
        for i in range(len(df)):
            current_price = df['close'].iloc[i]
            current_date = df.index[i]
            
            # Long entry
            if df['long_entry'].iloc[i] and position == 0:
                position = 1
                entry_price = current_price
                entry_date = current_date
                
            # Short entry
            elif df['short_entry'].iloc[i] and position == 0:
                position = -1
                entry_price = current_price
                entry_date = current_date
                
            # Long exit
            elif df['long_exit'].iloc[i] and position == 1:
                pnl = current_price - entry_price
                pnl_pct = (pnl / entry_price) * 100
                commission_cost = entry_price * self.commission * 2  # Entry + Exit
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
                    'capital': capital
                })
                
                position = 0
                
            # Short exit
            elif df['short_exit'].iloc[i] and position == -1:
                pnl = entry_price - current_price
                pnl_pct = (pnl / entry_price) * 100
                commission_cost = entry_price * self.commission * 2  # Entry + Exit
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
                    'capital': capital
                })
                
                position = 0
            
            results['equity_curve'].append({
                'date': current_date,
                'capital': capital,
                'position': position
            })
        
        # Calculate statistics
        results['statistics'] = self.calculate_statistics(results['trades'])
        
        return results
    
    def calculate_statistics(self, trades: List[Dict]) -> Dict:
        """Calculate comprehensive trading statistics"""
        if not trades:
            return {}
        
        df_trades = pd.DataFrame(trades)
        
        # Basic statistics
        total_trades = len(trades)
        winning_trades = df_trades[df_trades['net_pnl'] > 0]
        losing_trades = df_trades[df_trades['net_pnl'] <= 0]
        
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        
        total_pnl = df_trades['net_pnl'].sum()
        total_return = (df_trades['capital'].iloc[-1] - self.initial_capital) / self.initial_capital * 100
        
        avg_win = winning_trades['net_pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['net_pnl'].mean() if len(losing_trades) > 0 else 0
        
        # Profit factor
        total_wins = winning_trades['net_pnl'].sum() if len(winning_trades) > 0 else 0
        total_losses = abs(losing_trades['net_pnl'].sum()) if len(losing_trades) > 0 else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Drawdown calculation
        equity_curve = df_trades['capital'].values
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        # Sharpe ratio (simplified)
        returns = df_trades['net_pnl'] / self.initial_capital
        if returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
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
            'final_capital': df_trades['capital'].iloc[-1] if len(df_trades) > 0 else self.initial_capital
        }

def main():
    """Main execution function"""
    # Load data
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
    
    # Initialize ML classifier
    print("🤖 Initializing Lorentzian ML Classifier...")
    classifier = LorentzianClassifier(
        neighbors_count=8,
        max_bars_back=2000,
        feature_count=5,
        use_volatility_filter=True,
        use_regime_filter=True,
        use_adx_filter=False,
        use_ema_filter=False,
        use_sma_filter=False
    )
    
    # Generate signals
    print("📊 Generating ML signals...")
    df_signals = classifier.generate_signals(df)
    
    # Run backtest
    print("💰 Running backtest...")
    backtest_engine = BacktestEngine(initial_capital=10000, commission=0.001)
    results = backtest_engine.run_backtest(df_signals)
    
    # Print results
    print("\n" + "="*60)
    print("🏆 LORENTZIAN ML BACKTEST RESULTS")
    print("="*60)
    
    stats = results['statistics']
    print(f"📊 Total Trades: {stats.get('total_trades', 0)}")
    print(f"🏆 Win Rate: {stats.get('win_rate', 0):.2f}%")
    print(f"💰 Total Return: {stats.get('total_return', 0):.2f}%")
    print(f"💵 Final Capital: ${stats.get('final_capital', 0):,.2f}")
    print(f"⚡ Profit Factor: {stats.get('profit_factor', 0):.2f}")
    print(f"📊 Sharpe Ratio: {stats.get('sharpe_ratio', 0):.2f}")
    print(f"📉 Max Drawdown: {stats.get('max_drawdown', 0):.2f}%")
    print(f"🎪 Average Win: ${stats.get('avg_win', 0):.2f}")
    print(f"⚠️ Average Loss: ${stats.get('avg_loss', 0):.2f}")
    
    # Export results
    if results['trades']:
        trades_df = pd.DataFrame(results['trades'])
        trades_df.to_csv('lorentzian_ml_trades.csv', index=False)
        print(f"\n📁 Trades exported to: lorentzian_ml_trades.csv")
        
        # Create equity curve
        equity_df = pd.DataFrame(results['equity_curve'])
        equity_df.to_csv('lorentzian_ml_equity.csv', index=False)
        print(f"📁 Equity curve exported to: lorentzian_ml_equity.csv")
    
    # Plot results
    try:
        plt.figure(figsize=(15, 10))
        
        # Plot price and signals
        plt.subplot(2, 1, 1)
        plt.plot(df_signals.index, df_signals['close'], label='Price', alpha=0.7)
        
        # Plot entry signals
        long_entries = df_signals[df_signals['long_entry']]
        short_entries = df_signals[df_signals['short_entry']]
        
        plt.scatter(long_entries.index, long_entries['close'], 
                   color='green', marker='^', s=100, label='Long Entry')
        plt.scatter(short_entries.index, short_entries['close'], 
                   color='red', marker='v', s=100, label='Short Entry')
        
        plt.title('Lorentzian ML Classification - Trading Signals')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot equity curve
        plt.subplot(2, 1, 2)
        if results['equity_curve']:
            equity_df = pd.DataFrame(results['equity_curve'])
            plt.plot(equity_df['date'], equity_df['capital'], 
                    label='Equity Curve', color='blue')
            plt.axhline(y=backtest_engine.initial_capital, 
                       color='red', linestyle='--', alpha=0.5, label='Initial Capital')
            plt.title('Equity Curve')
            plt.xlabel('Date')
            plt.ylabel('Capital ($)')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('lorentzian_ml_results.png', dpi=300, bbox_inches='tight')
        print(f"📊 Chart saved to: lorentzian_ml_results.png")
        
    except Exception as e:
        print(f"⚠️ Error creating plots: {e}")

if __name__ == "__main__":
    main()