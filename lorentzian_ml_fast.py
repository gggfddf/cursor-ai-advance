#!/usr/bin/env python3
"""
Fast Lorentzian Classification ML Backtesting System
Optimized for speed and performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class FastTechnicalIndicators:
    """Fast technical indicators using vectorized operations"""
    
    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Fast RSI calculation"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """Fast CCI calculation"""
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        return (tp - sma_tp) / (0.015 * mad)
    
    @staticmethod
    def wt(hlc3: pd.Series, period1: int = 10, period2: int = 21) -> pd.Series:
        """Fast Wave Trend calculation"""
        esa = hlc3.ewm(span=period1).mean()
        d = (hlc3 - esa).abs().ewm(span=period1).mean()
        ci = (hlc3 - esa) / (0.015 * d)
        return ci.ewm(span=period2).mean()
    
    @staticmethod
    def momentum(prices: pd.Series, period: int = 10) -> pd.Series:
        """Fast momentum calculation"""
        return prices.pct_change(period) * 100
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Fast ATR calculation"""
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

class FastLorentzianClassifier:
    """Fast Lorentzian Classification ML System"""
    
    def __init__(self, 
                 neighbors_count: int = 8,
                 max_bars_back: int = 500,  # Reduced for speed
                 feature_count: int = 4,    # Reduced for speed
                 prediction_threshold: float = 0.6):
        
        self.neighbors_count = neighbors_count
        self.max_bars_back = max_bars_back
        self.feature_count = feature_count
        self.prediction_threshold = prediction_threshold
        
        self.indicators = FastTechnicalIndicators()
        
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate core features efficiently"""
        data = df.copy()
        data['hlc3'] = (data['high'] + data['low'] + data['close']) / 3
        
        print("📊 Calculating features...")
        
        # Core features for speed
        data['f1'] = self.indicators.rsi(data['close'], 14)
        data['f2'] = self.indicators.cci(data['high'], data['low'], data['close'], 20)
        data['f3'] = self.indicators.wt(data['hlc3'], 10, 21)
        data['f4'] = self.indicators.momentum(data['close'], 10)
        
        # Simple normalization to [-1, 1]
        for i in range(1, self.feature_count + 1):
            feature_col = f'f{i}'
            if feature_col in data.columns:
                data[feature_col] = self.normalize_feature(data[feature_col])
        
        return data
    
    def normalize_feature(self, feature: pd.Series) -> pd.Series:
        """Fast normalization"""
        feature_clean = feature.dropna()
        if len(feature_clean) == 0:
            return feature
        
        q25, q75 = feature_clean.quantile([0.25, 0.75])
        if q75 == q25:
            return pd.Series(0, index=feature.index)
        
        # Robust normalization using IQR
        normalized = (feature - feature_clean.median()) / (q75 - q25)
        return np.clip(normalized, -1, 1)
    
    def get_lorentzian_distance(self, current_features: np.ndarray, historical_features: np.ndarray) -> float:
        """Vectorized Lorentzian distance calculation"""
        return np.sum(np.log(1 + np.abs(current_features - historical_features)))
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate ML signals with optimized performance"""
        data = self.calculate_features(df)
        
        # Simple future return prediction
        data['y_train'] = np.where(
            data['close'].shift(-2) > data['close'] * 1.001, 1,
            np.where(data['close'].shift(-2) < data['close'] * 0.999, -1, 0)
        )
        
        # Pre-compute feature matrix for vectorization
        feature_cols = [f'f{i}' for i in range(1, self.feature_count + 1)]
        feature_matrix = data[feature_cols].values
        
        predictions = np.zeros(len(data))
        
        print("🤖 Generating ML signals...")
        
        # Optimized ML loop with progress tracking
        warmup_period = 100
        
        for i in range(warmup_period, len(data)):
            if i % 500 == 0:  # Progress indicator
                print(f"   Processing: {i}/{len(data)} ({i/len(data)*100:.1f}%)")
            
            current_features = feature_matrix[i]
            
            # Skip if any NaN values
            if np.isnan(current_features).any():
                continue
            
            # Efficient lookback window
            start_idx = max(0, i - self.max_bars_back)
            end_idx = i
            
            # Get historical features and labels
            historical_features = feature_matrix[start_idx:end_idx:3]  # Every 3rd bar for speed
            historical_labels = data['y_train'].iloc[start_idx:end_idx:3].values
            
            # Remove NaN rows
            valid_mask = ~np.isnan(historical_features).any(axis=1)
            historical_features = historical_features[valid_mask]
            historical_labels = historical_labels[valid_mask]
            
            if len(historical_features) < self.neighbors_count:
                continue
            
            # Vectorized distance calculation
            distances = np.array([
                self.get_lorentzian_distance(current_features, hist_feat)
                for hist_feat in historical_features
            ])
            
            # Get k nearest neighbors
            nearest_indices = np.argsort(distances)[:self.neighbors_count]
            nearest_labels = historical_labels[nearest_indices]
            
            # Weighted prediction
            nearest_distances = distances[nearest_indices]
            weights = 1 / (1 + nearest_distances)
            predictions[i] = np.sum(weights * nearest_labels) / np.sum(weights)
        
        data['prediction'] = predictions
        
        # Simple signal generation
        data['signal'] = np.where(
            data['prediction'] > self.prediction_threshold, 1,
            np.where(data['prediction'] < -self.prediction_threshold, -1, 0)
        )
        
        # Entry/exit signals
        data['signal_change'] = data['signal'].diff()
        data['long_entry'] = (data['signal'] == 1) & (data['signal_change'] != 0)
        data['short_entry'] = (data['signal'] == -1) & (data['signal_change'] != 0)
        
        # Simple exit after 3 bars
        data['long_exit'] = data['long_entry'].shift(3).fillna(False)
        data['short_exit'] = data['short_entry'].shift(3).fillna(False)
        
        print("✅ Signal generation complete!")
        return data

class FastBacktestEngine:
    """Fast backtesting engine"""
    
    def __init__(self, initial_capital: float = 10000, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        
    def run_backtest(self, df: pd.DataFrame) -> Dict:
        """Run fast backtest"""
        print("💰 Running backtest...")
        
        trades = []
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
                commission_cost = entry_price * self.commission * 2
                net_pnl = pnl - commission_cost
                capital += net_pnl
                
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': current_date,
                    'type': 'long',
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl': pnl,
                    'pnl_pct': (pnl / entry_price) * 100,
                    'net_pnl': net_pnl,
                    'capital': capital
                })
                
                position = 0
                
            # Short exit
            elif df['short_exit'].iloc[i] and position == -1:
                pnl = entry_price - current_price
                commission_cost = entry_price * self.commission * 2
                net_pnl = pnl - commission_cost
                capital += net_pnl
                
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': current_date,
                    'type': 'short',
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl': pnl,
                    'pnl_pct': (pnl / entry_price) * 100,
                    'net_pnl': net_pnl,
                    'capital': capital
                })
                
                position = 0
        
        # Calculate statistics
        stats = self.calculate_statistics(trades)
        
        return {
            'trades': trades,
            'statistics': stats
        }
    
    def calculate_statistics(self, trades: List[Dict]) -> Dict:
        """Calculate key statistics"""
        if not trades:
            return {}
        
        df_trades = pd.DataFrame(trades)
        
        total_trades = len(trades)
        winning_trades = df_trades[df_trades['net_pnl'] > 0]
        losing_trades = df_trades[df_trades['net_pnl'] <= 0]
        
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        total_return = (df_trades['capital'].iloc[-1] - self.initial_capital) / self.initial_capital * 100
        
        avg_win = winning_trades['net_pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['net_pnl'].mean() if len(losing_trades) > 0 else 0
        
        total_wins = winning_trades['net_pnl'].sum() if len(winning_trades) > 0 else 0
        total_losses = abs(losing_trades['net_pnl'].sum()) if len(losing_trades) > 0 else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Drawdown
        equity_curve = df_trades['capital'].values
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        # Sharpe ratio
        returns = df_trades['net_pnl'] / self.initial_capital
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
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
    print("🚀 Fast Lorentzian ML Backtesting System")
    print("=" * 50)
    
    # Load data
    print("📁 Loading XAUUSD data...")
    try:
        df = pd.read_csv('XAU_1d_data_clean.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.columns = [col.lower() for col in df.columns]
        print(f"✅ Data loaded: {len(df)} rows")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Initialize fast classifier
    print("\n🤖 Initializing Fast ML Classifier...")
    classifier = FastLorentzianClassifier(
        neighbors_count=8,
        max_bars_back=500,  # Reduced for speed
        feature_count=4,    # Reduced for speed
        prediction_threshold=0.6
    )
    
    # Generate signals
    print("\n📊 Generating signals...")
    df_signals = classifier.generate_signals(df)
    
    # Run backtest
    print("\n💰 Running backtest...")
    backtest_engine = FastBacktestEngine(initial_capital=10000, commission=0.001)
    results = backtest_engine.run_backtest(df_signals)
    
    # Display results
    print("\n" + "=" * 60)
    print("🏆 FAST LORENTZIAN ML BACKTEST RESULTS")
    print("=" * 60)
    
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
        trades_df.to_csv('fast_lorentzian_trades.csv', index=False)
        print(f"\n📁 Trades exported to: fast_lorentzian_trades.csv")
        
        # Quick plot
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(trades_df['exit_date'], trades_df['capital'], 
                    label='Equity Curve', linewidth=2)
            plt.axhline(y=10000, color='red', linestyle='--', 
                       alpha=0.7, label='Initial Capital')
            plt.title('Fast Lorentzian ML - Equity Curve')
            plt.xlabel('Date')
            plt.ylabel('Capital ($)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('fast_lorentzian_equity.png', dpi=300, bbox_inches='tight')
            print(f"📊 Chart saved to: fast_lorentzian_equity.png")
        except Exception as e:
            print(f"⚠️ Error creating plot: {e}")
    
    print("\n✅ Fast backtesting complete!")

if __name__ == "__main__":
    main()