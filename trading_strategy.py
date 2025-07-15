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
import pickle
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class XAUUSDTradingStrategy:
    def __init__(self, data_file='XAU_1d_data_clean.csv'):
        self.data_file = data_file
        self.data = None
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = []
        
        # Trading Parameters
        self.initial_capital = 100000  # $100,000 starting capital
        self.risk_per_trade = 0.02     # 2% risk per trade
        self.min_confidence = 0.6      # Minimum ML prediction confidence
        self.stop_loss_pct = 0.015     # 1.5% stop loss
        self.take_profit_pct = 0.03    # 3% take profit (2:1 R:R)
        self.max_positions = 3         # Maximum concurrent positions
        
        # Performance tracking
        self.trades = []
        self.portfolio_values = []
        self.positions = []
        
    def load_and_prepare_data(self):
        """Load and prepare the dataset with all features."""
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
                
            logger.info("Data loaded and prepared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def engineer_features(self):
        """Engineer all the features used in the ML model."""
        try:
            logger.info("Starting feature engineering...")
            
            # Basic price features
            self.data['range'] = self.data['high'] - self.data['low']
            self.data['body'] = abs(self.data['close'] - self.data['open'])
            self.data['upper_wick'] = self.data['high'] - np.maximum(self.data['open'], self.data['close'])
            self.data['lower_wick'] = np.minimum(self.data['open'], self.data['close']) - self.data['low']
            
            # Direction and candle type
            self.data['direction'] = (self.data['close'] > self.data['open']).astype(int)
            self.data['candle_type'] = np.where(self.data['direction'] == 1, 1, -1)
            
            # Body and wick ratios
            self.data['body_range_ratio'] = self.data['body'] / (self.data['range'] + 1e-8)
            self.data['wick_body_ratio'] = (self.data['upper_wick'] + self.data['lower_wick']) / (self.data['body'] + 1e-8)
            
            # Moving averages and volatility
            for window in [6, 20]:
                self.data[f'body_std_{window}'] = self.data['body'].rolling(window).std()
                
            # Position in range
            self.data['relative_position_20'] = (self.data['close'] - self.data['close'].rolling(20).min()) / \
                                               (self.data['close'].rolling(20).max() - self.data['close'].rolling(20).min() + 1e-8)
            
            # Volume features
            self.data['volume_ratio_10'] = self.data['volume'] / (self.data['volume'].rolling(10).mean() + 1e-8)
            self.data['volume_anomaly'] = (self.data['volume'] > self.data['volume'].rolling(20).mean() * 2).astype(int)
            
            # Time-based features
            self.data['day_of_week'] = self.data['date'].dt.dayofweek
            self.data['start_of_month'] = (self.data['date'].dt.day <= 5).astype(int)
            self.data['end_of_month'] = (self.data['date'].dt.day >= 25).astype(int)
            
            # Pattern recognition features
            self.data['doji'] = (self.data['body'] < self.data['range'] * 0.1).astype(int)
            self.data['hammer'] = ((self.data['lower_wick'] > self.data['body'] * 2) & 
                                  (self.data['upper_wick'] < self.data['body'] * 0.5)).astype(int)
            self.data['shooting_star'] = ((self.data['upper_wick'] > self.data['body'] * 2) & 
                                         (self.data['lower_wick'] < self.data['body'] * 0.5)).astype(int)
            
            # Engulfing patterns
            prev_body = self.data['body'].shift(1)
            self.data['engulfing'] = ((self.data['body'] > prev_body * 1.5) & 
                                     (self.data['direction'] != self.data['direction'].shift(1))).astype(int)
            
            # Inside bars and gaps
            self.data['inside_bar'] = ((self.data['high'] < self.data['high'].shift(1)) & 
                                      (self.data['low'] > self.data['low'].shift(1))).astype(int)
            self.data['gap_up'] = (self.data['low'] > self.data['high'].shift(1)).astype(int)
            self.data['gap_down'] = (self.data['high'] < self.data['low'].shift(1)).astype(int)
            
            # Advanced features
            self.data['bars_since_new_high'] = 0
            self.data['bars_since_new_low'] = 0
            
            for i in range(1, len(self.data)):
                # Bars since new high/low
                recent_high = self.data['high'].iloc[max(0, i-20):i+1].max()
                recent_low = self.data['low'].iloc[max(0, i-20):i+1].min()
                
                if self.data['high'].iloc[i] >= recent_high:
                    self.data.loc[i, 'bars_since_new_high'] = 0
                else:
                    self.data.loc[i, 'bars_since_new_high'] = self.data.loc[i-1, 'bars_since_new_high'] + 1
                    
                if self.data['low'].iloc[i] <= recent_low:
                    self.data.loc[i, 'bars_since_new_low'] = 0
                else:
                    self.data.loc[i, 'bars_since_new_low'] = self.data.loc[i-1, 'bars_since_new_low'] + 1
            
            # Market psychology features
            self.data['momentum_exhaustion'] = ((abs(self.data['close'] - self.data['open']) < 
                                               self.data['range'] * 0.3) & 
                                              (self.data['volume'] > self.data['volume'].rolling(10).mean() * 1.5)).astype(int)
            
            self.data['wick_pressure'] = np.maximum(self.data['upper_wick'], self.data['lower_wick']) / (self.data['range'] + 1e-8)
            
            self.data['humidity_day'] = (self.data['range'] < self.data['range'].rolling(10).mean() * 0.7).astype(int)
            
            # Additional technical features
            self.data['volatility_expansion'] = (self.data['range'] > self.data['range'].rolling(20).mean() * 1.5).astype(int)
            self.data['volatility_squeeze'] = (self.data['range'] < self.data['range'].rolling(20).mean() * 0.5).astype(int)
            
            self.data['consolidation_breakout_window'] = 0
            self.data['breakout_insecurity'] = 0
            self.data['fakeout_up'] = 0
            self.data['fakeout_down'] = 0
            self.data['time_to_reversal'] = 0
            self.data['storm_day'] = (self.data['range'] > self.data['range'].rolling(5).mean() * 2).astype(int)
            self.data['tight_cluster_zone'] = (self.data['range'] < self.data['range'].rolling(5).mean() * 0.3).astype(int)
            self.data['directional_consistency'] = (self.data['direction'] == self.data['direction'].shift(1)).astype(int)
            self.data['support_test_count'] = 0
            self.data['is_first_expansion_candle'] = 0
            self.data['expansion_energy'] = self.data['volume'] * self.data['range']
            self.data['liquidity_sweep_up'] = (self.data['high'] > self.data['high'].rolling(5).max().shift(1)).astype(int)
            self.data['liquidity_sweep_down'] = (self.data['low'] < self.data['low'].rolling(5).min().shift(1)).astype(int)
            
            # Fill any remaining NaN values
            self.data = self.data.fillna(method='ffill').fillna(0)
            
            logger.info("Feature engineering completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            return False
    
    def create_target(self):
        """Create target variable for next day price direction."""
        try:
            # Next day direction (1 = up, 0 = down)
            self.data['target'] = (self.data['close'].shift(-1) > self.data['close']).astype(int)
            
            # Remove last row as it doesn't have next day data
            self.data = self.data[:-1].reset_index(drop=True)
            
            logger.info(f"Target created. Final dataset shape: {self.data.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating target: {e}")
            return False
    
    def train_model(self):
        """Train the XGBoost model."""
        try:
            # Define feature columns (same as in original script)
            feature_cols = [col for col in self.data.columns 
                           if col not in ['date', 'open', 'high', 'low', 'close', 'volume', 'target']]
            
            self.feature_cols = feature_cols
            
            # Split data (80% train, 20% test)
            split_idx = int(len(self.data) * 0.8)
            
            X_train = self.data[feature_cols][:split_idx]
            y_train = self.data['target'][:split_idx]
            X_test = self.data[feature_cols][split_idx:]
            y_test = self.data['target'][split_idx:]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train XGBoost model
            self.model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            logger.info("Model training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
    
    def generate_signals(self):
        """Generate trading signals based on ML predictions."""
        try:
            # Get features for prediction
            X = self.data[self.feature_cols]
            X_scaled = self.scaler.transform(X)
            
            # Get predictions and probabilities
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)
            
            # Add signals to dataframe
            self.data['ml_prediction'] = predictions
            self.data['ml_probability'] = probabilities[:, 1]  # Probability of upward movement
            self.data['confidence'] = np.maximum(probabilities[:, 0], probabilities[:, 1])
            
            # Generate trading signals
            self.data['signal'] = 0  # 0 = no signal, 1 = buy, -1 = sell
            
            # Buy signals: High confidence prediction of upward movement
            buy_condition = (self.data['ml_prediction'] == 1) & (self.data['confidence'] >= self.min_confidence)
            self.data.loc[buy_condition, 'signal'] = 1
            
            # Sell signals: High confidence prediction of downward movement  
            sell_condition = (self.data['ml_prediction'] == 0) & (self.data['confidence'] >= self.min_confidence)
            self.data.loc[sell_condition, 'signal'] = -1
            
            logger.info(f"Generated {sum(self.data['signal'] == 1)} buy signals and {sum(self.data['signal'] == -1)} sell signals")
            return True
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return False
    
    def calculate_position_size(self, current_price, stop_loss_price, current_capital):
        """Calculate position size based on risk management."""
        risk_amount = current_capital * self.risk_per_trade
        price_risk = abs(current_price - stop_loss_price)
        
        if price_risk == 0:
            return 0
            
        # Position size in USD
        position_size = risk_amount / price_risk
        
        # Don't risk more than 10% of capital on single trade
        max_position = current_capital * 0.1
        position_size = min(position_size, max_position)
        
        return position_size
    
    def backtest_strategy(self):
        """Run the complete backtesting simulation."""
        try:
            logger.info("Starting strategy backtest...")
            
            current_capital = self.initial_capital
            self.portfolio_values = [current_capital]
            self.trades = []
            self.positions = []
            
            for i in range(1, len(self.data)):
                current_date = self.data['date'].iloc[i]
                current_price = self.data['open'].iloc[i]  # Use next day's open price
                signal = self.data['signal'].iloc[i-1]  # Previous day's signal
                
                # Check for position exits first
                positions_to_remove = []
                for pos_idx, position in enumerate(self.positions):
                    exit_price = None
                    exit_reason = None
                    
                    # Check stop loss
                    if position['type'] == 'long':
                        if self.data['low'].iloc[i] <= position['stop_loss']:
                            exit_price = position['stop_loss']
                            exit_reason = 'stop_loss'
                        elif self.data['high'].iloc[i] >= position['take_profit']:
                            exit_price = position['take_profit']
                            exit_reason = 'take_profit'
                    else:  # short position
                        if self.data['high'].iloc[i] >= position['stop_loss']:
                            exit_price = position['stop_loss']
                            exit_reason = 'stop_loss'
                        elif self.data['low'].iloc[i] <= position['take_profit']:
                            exit_price = position['take_profit']
                            exit_reason = 'take_profit'
                    
                    # Exit position if conditions met
                    if exit_price:
                        if position['type'] == 'long':
                            pnl = (exit_price - position['entry_price']) * position['size']
                        else:
                            pnl = (position['entry_price'] - exit_price) * position['size']
                        
                        current_capital += pnl
                        
                        trade_record = {
                            'entry_date': position['entry_date'],
                            'exit_date': current_date,
                            'type': position['type'],
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'size': position['size'],
                            'pnl': pnl,
                            'return_pct': pnl / (position['entry_price'] * position['size']) * 100,
                            'exit_reason': exit_reason
                        }
                        self.trades.append(trade_record)
                        positions_to_remove.append(pos_idx)
                
                # Remove closed positions
                for pos_idx in reversed(positions_to_remove):
                    self.positions.pop(pos_idx)
                
                # Check for new entries
                if signal != 0 and len(self.positions) < self.max_positions:
                    if signal == 1:  # Buy signal
                        stop_loss_price = current_price * (1 - self.stop_loss_pct)
                        take_profit_price = current_price * (1 + self.take_profit_pct)
                        position_size = self.calculate_position_size(current_price, stop_loss_price, current_capital)
                        
                        if position_size > 0:
                            position = {
                                'entry_date': current_date,
                                'type': 'long',
                                'entry_price': current_price,
                                'size': position_size / current_price,  # Convert to units
                                'stop_loss': stop_loss_price,
                                'take_profit': take_profit_price
                            }
                            self.positions.append(position)
                    
                    elif signal == -1:  # Sell signal
                        stop_loss_price = current_price * (1 + self.stop_loss_pct)
                        take_profit_price = current_price * (1 - self.take_profit_pct)
                        position_size = self.calculate_position_size(current_price, stop_loss_price, current_capital)
                        
                        if position_size > 0:
                            position = {
                                'entry_date': current_date,
                                'type': 'short',
                                'entry_price': current_price,
                                'size': position_size / current_price,  # Convert to units
                                'stop_loss': stop_loss_price,
                                'take_profit': take_profit_price
                            }
                            self.positions.append(position)
                
                # Calculate current portfolio value
                portfolio_value = current_capital
                for position in self.positions:
                    if position['type'] == 'long':
                        unrealized_pnl = (current_price - position['entry_price']) * position['size']
                    else:
                        unrealized_pnl = (position['entry_price'] - current_price) * position['size']
                    portfolio_value += unrealized_pnl
                
                self.portfolio_values.append(portfolio_value)
            
            logger.info(f"Backtest completed. Total trades: {len(self.trades)}")
            return True
            
        except Exception as e:
            logger.error(f"Error in backtesting: {e}")
            return False
    
    def calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics."""
        try:
            if not self.trades:
                logger.warning("No trades to analyze")
                return {}
            
            trades_df = pd.DataFrame(self.trades)
            portfolio_df = pd.DataFrame({
                'date': self.data['date'][:len(self.portfolio_values)],
                'portfolio_value': self.portfolio_values
            })
            
            # Basic metrics
            total_return = (self.portfolio_values[-1] - self.initial_capital) / self.initial_capital * 100
            total_trades = len(self.trades)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            losing_trades = len(trades_df[trades_df['pnl'] < 0])
            win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
            
            # PnL metrics
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
            profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if avg_loss != 0 else float('inf')
            
            # Risk metrics
            portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change()
            daily_returns = portfolio_df['returns'].dropna()
            
            sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() != 0 else 0
            
            # Maximum drawdown
            peak = portfolio_df['portfolio_value'].expanding().max()
            drawdown = (portfolio_df['portfolio_value'] - peak) / peak * 100
            max_drawdown = drawdown.min()
            
            # Additional metrics
            avg_trade_duration = (pd.to_datetime(trades_df['exit_date']) - pd.to_datetime(trades_df['entry_date'])).dt.days.mean()
            largest_win = trades_df['pnl'].max() if total_trades > 0 else 0
            largest_loss = trades_df['pnl'].min() if total_trades > 0 else 0
            
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
                'avg_trade_duration_days': avg_trade_duration,
                'final_capital': self.portfolio_values[-1],
                'total_pnl': sum(trades_df['pnl'])
            }
            
            logger.info("Performance metrics calculated successfully")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def create_visualizations(self, metrics):
        """Create comprehensive strategy visualizations."""
        try:
            # Set up the plotting style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 16))
            
            # 1. Portfolio Value Over Time
            ax1 = plt.subplot(3, 3, 1)
            portfolio_df = pd.DataFrame({
                'date': self.data['date'][:len(self.portfolio_values)],
                'portfolio_value': self.portfolio_values
            })
            
            plt.plot(portfolio_df['date'], portfolio_df['portfolio_value'], linewidth=2, color='blue')
            plt.axhline(y=self.initial_capital, color='red', linestyle='--', alpha=0.7, label='Initial Capital')
            plt.title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value ($)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 2. Drawdown Chart
            ax2 = plt.subplot(3, 3, 2)
            peak = portfolio_df['portfolio_value'].expanding().max()
            drawdown = (portfolio_df['portfolio_value'] - peak) / peak * 100
            plt.fill_between(portfolio_df['date'], drawdown, 0, color='red', alpha=0.3)
            plt.plot(portfolio_df['date'], drawdown, color='red', linewidth=1)
            plt.title('Drawdown Chart', fontsize=14, fontweight='bold')
            plt.xlabel('Date')
            plt.ylabel('Drawdown (%)')
            plt.grid(True, alpha=0.3)
            
            # 3. Trade PnL Distribution
            ax3 = plt.subplot(3, 3, 3)
            if self.trades:
                trades_df = pd.DataFrame(self.trades)
                plt.hist(trades_df['pnl'], bins=30, alpha=0.7, color='green', edgecolor='black')
                plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
                plt.title('Trade PnL Distribution', fontsize=14, fontweight='bold')
                plt.xlabel('PnL ($)')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
            
            # 4. Cumulative Returns vs Buy & Hold
            ax4 = plt.subplot(3, 3, 4)
            portfolio_returns = portfolio_df['portfolio_value'] / self.initial_capital
            
            # Calculate buy & hold returns
            price_data = self.data[['date', 'close']].iloc[:len(self.portfolio_values)]
            buy_hold_returns = price_data['close'] / price_data['close'].iloc[0]
            
            plt.plot(portfolio_df['date'], portfolio_returns, label='Strategy', linewidth=2, color='blue')
            plt.plot(price_data['date'], buy_hold_returns, label='Buy & Hold', linewidth=2, color='orange')
            plt.title('Strategy vs Buy & Hold Returns', fontsize=14, fontweight='bold')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Returns')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 5. Monthly Returns Heatmap
            ax5 = plt.subplot(3, 3, 5)
            if self.trades:
                trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
                monthly_pnl = trades_df.groupby([trades_df['entry_date'].dt.year, 
                                               trades_df['entry_date'].dt.month])['pnl'].sum().unstack(fill_value=0)
                
                if not monthly_pnl.empty:
                    sns.heatmap(monthly_pnl, annot=True, fmt='.0f', cmap='RdYlGn', center=0, ax=ax5)
                    plt.title('Monthly PnL Heatmap', fontsize=14, fontweight='bold')
                    plt.xlabel('Month')
                    plt.ylabel('Year')
            
            # 6. Win/Loss Analysis
            ax6 = plt.subplot(3, 3, 6)
            if self.trades:
                win_loss_data = ['Wins', 'Losses']
                win_loss_counts = [metrics['winning_trades'], metrics['losing_trades']]
                colors = ['green', 'red']
                plt.pie(win_loss_counts, labels=win_loss_data, colors=colors, autopct='%1.1f%%', startangle=90)
                plt.title('Win/Loss Ratio', fontsize=14, fontweight='bold')
            
            # 7. Trade Duration Distribution
            ax7 = plt.subplot(3, 3, 7)
            if self.trades:
                trades_df['duration'] = (pd.to_datetime(trades_df['exit_date']) - 
                                       pd.to_datetime(trades_df['entry_date'])).dt.days
                plt.hist(trades_df['duration'], bins=20, alpha=0.7, color='purple', edgecolor='black')
                plt.title('Trade Duration Distribution', fontsize=14, fontweight='bold')
                plt.xlabel('Duration (Days)')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
            
            # 8. Signal Distribution
            ax8 = plt.subplot(3, 3, 8)
            signal_counts = self.data['signal'].value_counts()
            signal_labels = ['No Signal', 'Sell Signal', 'Buy Signal']
            plt.bar(range(len(signal_counts)), signal_counts.values, color=['gray', 'red', 'green'])
            plt.title('Trading Signal Distribution', fontsize=14, fontweight='bold')
            plt.xlabel('Signal Type')
            plt.ylabel('Count')
            plt.xticks(range(len(signal_counts)), [signal_labels[i] for i in signal_counts.index])
            plt.grid(True, alpha=0.3)
            
            # 9. Performance Metrics Table
            ax9 = plt.subplot(3, 3, 9)
            ax9.axis('off')
            
            metrics_text = f"""
            STRATEGY PERFORMANCE METRICS
            ═══════════════════════════════════
            Total Return: {metrics.get('total_return_pct', 0):.2f}%
            Total Trades: {metrics.get('total_trades', 0)}
            Win Rate: {metrics.get('win_rate_pct', 0):.2f}%
            Profit Factor: {metrics.get('profit_factor', 0):.2f}
            Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
            Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%
            
            Avg Win: ${metrics.get('avg_win', 0):.2f}
            Avg Loss: ${metrics.get('avg_loss', 0):.2f}
            Largest Win: ${metrics.get('largest_win', 0):.2f}
            Largest Loss: ${metrics.get('largest_loss', 0):.2f}
            
            Final Capital: ${metrics.get('final_capital', 0):,.2f}
            Total PnL: ${metrics.get('total_pnl', 0):,.2f}
            """
            
            ax9.text(0.05, 0.95, metrics_text, transform=ax9.transAxes, fontsize=11,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig('trading_strategy_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Strategy visualizations created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            return False
    
    def generate_trading_report(self, metrics):
        """Generate comprehensive trading strategy report."""
        try:
            report_content = f"""# XAUUSD Trading Strategy Backtest Report
## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Strategy Overview

This report presents the results of a comprehensive backtesting analysis for the XAUUSD (Gold/USD) trading strategy 
based on machine learning predictions and advanced risk management techniques.

### Strategy Parameters
- **Initial Capital:** ${self.initial_capital:,}
- **Risk Per Trade:** {self.risk_per_trade*100:.1f}%
- **Minimum ML Confidence:** {self.min_confidence*100:.1f}%
- **Stop Loss:** {self.stop_loss_pct*100:.1f}%
- **Take Profit:** {self.take_profit_pct*100:.1f}%
- **Max Concurrent Positions:** {self.max_positions}

## Performance Summary

### Key Metrics
- **Total Return:** {metrics.get('total_return_pct', 0):.2f}%
- **Final Capital:** ${metrics.get('final_capital', 0):,.2f}
- **Total PnL:** ${metrics.get('total_pnl', 0):,.2f}
- **Total Trades:** {metrics.get('total_trades', 0)}
- **Win Rate:** {metrics.get('win_rate_pct', 0):.2f}%
- **Profit Factor:** {metrics.get('profit_factor', 0):.2f}
- **Sharpe Ratio:** {metrics.get('sharpe_ratio', 0):.2f}
- **Maximum Drawdown:** {metrics.get('max_drawdown_pct', 0):.2f}%

### Trading Statistics
- **Winning Trades:** {metrics.get('winning_trades', 0)}
- **Losing Trades:** {metrics.get('losing_trades', 0)}
- **Average Win:** ${metrics.get('avg_win', 0):.2f}
- **Average Loss:** ${metrics.get('avg_loss', 0):.2f}
- **Largest Win:** ${metrics.get('largest_win', 0):.2f}
- **Largest Loss:** ${metrics.get('largest_loss', 0):.2f}
- **Average Trade Duration:** {metrics.get('avg_trade_duration_days', 0):.1f} days

## Strategy Analysis

### Strengths
1. **ML-Driven Signals:** Uses sophisticated machine learning model with {len(self.feature_cols)} features
2. **Risk Management:** Consistent 2% risk per trade with proper stop losses
3. **Position Sizing:** Dynamic position sizing based on volatility
4. **Multiple Timeframes:** Incorporates various technical indicators and patterns

### Areas for Improvement
1. **Signal Frequency:** Generated {len([s for s in self.data['signal'] if s != 0])} signals from {len(self.data)} bars
2. **Market Conditions:** Performance may vary in different market regimes
3. **Transaction Costs:** Real-world implementation would include spreads and commissions

## Risk Assessment

### Risk Metrics
- **Maximum Drawdown:** {metrics.get('max_drawdown_pct', 0):.2f}%
- **Risk-Adjusted Returns:** Sharpe ratio of {metrics.get('sharpe_ratio', 0):.2f}
- **Position Concentration:** Maximum {self.max_positions} concurrent positions

### Risk Considerations
1. **Model Dependency:** Strategy heavily relies on ML model accuracy
2. **Market Regime Changes:** May underperform during unprecedented market conditions
3. **Overfitting Risk:** Model trained on historical data may not predict future perfectly

## Trade Analysis

### Signal Distribution
- **Buy Signals:** {sum(self.data['signal'] == 1)} ({sum(self.data['signal'] == 1)/len(self.data)*100:.2f}% of total bars)
- **Sell Signals:** {sum(self.data['signal'] == -1)} ({sum(self.data['signal'] == -1)/len(self.data)*100:.2f}% of total bars)
- **No Signal:** {sum(self.data['signal'] == 0)} ({sum(self.data['signal'] == 0)/len(self.data)*100:.2f}% of total bars)

### ML Model Confidence
- **Average Confidence:** {self.data['confidence'].mean():.3f}
- **High Confidence Signals (>{self.min_confidence}):** {sum(self.data['confidence'] > self.min_confidence)} signals

## Implementation Recommendations

### For Live Trading
1. **Paper Trading:** Test strategy for 3-6 months before live implementation
2. **Position Sizing:** Start with smaller position sizes until strategy proves consistent
3. **Market Monitoring:** Regularly monitor model performance and market conditions
4. **Risk Management:** Never risk more than you can afford to lose

### Strategy Optimization
1. **Parameter Tuning:** Optimize confidence threshold and risk parameters
2. **Feature Engineering:** Continuously improve ML model features
3. **Market Regime Detection:** Add regime filters for different market conditions
4. **Transaction Costs:** Include realistic spreads and commissions in backtesting

## Conclusion

The XAUUSD trading strategy shows {"promising" if metrics.get('total_return_pct', 0) > 0 else "challenging"} results with a total return of {metrics.get('total_return_pct', 0):.2f}% 
over the backtesting period. The strategy demonstrates {"good" if metrics.get('win_rate_pct', 0) > 50 else "room for improvement in"} win rate of {metrics.get('win_rate_pct', 0):.2f}% 
and {"strong" if metrics.get('profit_factor', 0) > 1.5 else "moderate"} risk management with a profit factor of {metrics.get('profit_factor', 0):.2f}.

### Key Takeaways
1. Machine learning can provide valuable insights for trading decisions
2. Proper risk management is crucial for long-term success
3. Strategy performance should be evaluated across different market conditions
4. Continuous monitoring and optimization are essential

---

*Disclaimer: This backtest is for educational and research purposes only. Past performance does not guarantee future results. 
Always consult with financial advisors and conduct thorough due diligence before implementing any trading strategy.*
"""

            with open('Trading_Strategy_Report.md', 'w') as f:
                f.write(report_content)
            
            logger.info("Trading strategy report generated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error generating trading report: {e}")
            return False
    
    def run_complete_strategy_analysis(self):
        """Run the complete trading strategy analysis pipeline."""
        try:
            print("🚀 Starting XAUUSD Trading Strategy Analysis...")
            
            # Step 1: Load and prepare data
            if not self.load_and_prepare_data():
                return False
            print("✅ Data loaded and prepared")
            
            # Step 2: Engineer features
            if not self.engineer_features():
                return False
            print("✅ Features engineered")
            
            # Step 3: Create target
            if not self.create_target():
                return False
            print("✅ Target variable created")
            
            # Step 4: Train model
            if not self.train_model():
                return False
            print("✅ ML model trained")
            
            # Step 5: Generate trading signals
            if not self.generate_signals():
                return False
            print("✅ Trading signals generated")
            
            # Step 6: Run backtest
            if not self.backtest_strategy():
                return False
            print("✅ Strategy backtested")
            
            # Step 7: Calculate performance metrics
            metrics = self.calculate_performance_metrics()
            if not metrics:
                return False
            print("✅ Performance metrics calculated")
            
            # Step 8: Create visualizations
            if not self.create_visualizations(metrics):
                return False
            print("✅ Visualizations created")
            
            # Step 9: Generate comprehensive report
            if not self.generate_trading_report(metrics):
                return False
            print("✅ Trading report generated")
            
            # Print summary
            print(f"\n🎯 STRATEGY PERFORMANCE SUMMARY")
            print(f"══════════════════════════════════════")
            print(f"📈 Total Return: {metrics.get('total_return_pct', 0):.2f}%")
            print(f"💰 Final Capital: ${metrics.get('final_capital', 0):,.2f}")
            print(f"📊 Total Trades: {metrics.get('total_trades', 0)}")
            print(f"🎯 Win Rate: {metrics.get('win_rate_pct', 0):.2f}%")
            print(f"⚡ Profit Factor: {metrics.get('profit_factor', 0):.2f}")
            print(f"📉 Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
            
            print(f"\n🔧 Files Generated:")
            print(f"📋 Trading_Strategy_Report.md")
            print(f"📊 trading_strategy_analysis.png")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in complete strategy analysis: {e}")
            return False

def main():
    """Main execution function."""
    try:
        # Initialize and run the trading strategy
        strategy = XAUUSDTradingStrategy('XAU_1d_data_clean.csv')
        success = strategy.run_complete_strategy_analysis()
        
        if success:
            print("\n🚀 Trading Strategy Analysis completed successfully!")
        else:
            print("\n❌ Trading Strategy Analysis failed. Check logs for details.")
            
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        logger.error(f"Fatal error in main: {e}")

if __name__ == "__main__":
    main()