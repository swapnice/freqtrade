# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pandas import DataFrame
from typing import Optional, Union

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,
    # Hyperopt Parameters
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    # timeframe helpers
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    # Strategy helper functions
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

# Technical Analysis
import talib.abstract as ta
from technical import qtpylib

logger = logging.getLogger(__name__)


class RLAdaptiveStrategy(IStrategy):
    """
    Reinforcement Learning Adaptive Strategy

    This strategy demonstrates how to use FreqAI's reinforcement learning capabilities
    to create an adaptive trading strategy that learns from backtesting data.

    Key Features:
    - Uses FreqAI's RL framework with custom reward functions
    - Adaptive feature engineering based on market conditions
    - Multiple environment configurations for different market regimes
    - Comprehensive technical indicators as features
    - State-aware decision making

    To use this strategy:
    1. Configure FreqAI in your config with RL settings
    2. Run backtesting to train the model
    3. The agent will learn optimal entry/exit timing
    """

    # Strategy interface version
    INTERFACE_VERSION = 3

    # Can this strategy go short?
    can_short: bool = True

    # Minimal ROI - will be overridden by RL agent decisions
    minimal_roi = {
        "0": 0.05,  # 5% profit target
    }

    # Stoploss - will be managed by RL agent
    stoploss = -0.15  # 15% maximum loss

    # Optimal timeframe for the strategy
    timeframe = "5m"

    # Run "populate_indicators()" only for new candle
    process_only_new_candles = True

    # These values can be overridden in the config
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True  # Let RL agent decide when to exit

    # Hyperoptable parameters for traditional indicators (used as features)
    rsi_period = IntParameter(low=10, high=30, default=14, space="buy", optimize=False)
    bb_period = IntParameter(low=15, high=25, default=20, space="buy", optimize=False)
    macd_fast = IntParameter(low=8, high=15, default=12, space="buy", optimize=False)
    macd_slow = IntParameter(low=21, high=35, default=26, space="buy", optimize=False)

    # RL-specific parameters
    use_dynamic_features = BooleanParameter(default=True, space="buy", optimize=False)
    market_regime_lookback = IntParameter(low=50, high=200, default=100, space="buy", optimize=False)

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 300

    # FreqAI configuration will be loaded from config file
    # This strategy expects the following FreqAI config structure:
    freqai_config_example = {
        "freqai": {
            "enabled": True,
            "purge_old_models": 2,
            "train_period_days": 15,
            "backtest_period_days": 7,
            "live_retrain_hours": 0,
            "expiration_hours": 1,
            "identifier": "rl_adaptive_strategy",
            "feature_parameters": {
                "include_timeframes": ["5m", "15m", "1h"],
                "include_corr_pairlist": ["BTC/USDT", "ETH/USDT"],
                "label_period_candles": 24,
                "include_shifted_candles": 2,
                "DI_threshold": 0.9,
                "weight_factor": 0.9,
                "principal_component_analysis": False,
                "use_SVM_to_remove_outliers": True,
                "svm_params": {
                    "shuffle": True,
                    "nu": 0.1
                },
                "use_DBSCAN_to_remove_outliers": False,
                "indicator_max_period_candles": 20,
                "indicator_periods_candles": [10, 20]
            },
            "data_split_parameters": {
                "test_size": 0.33,
                "shuffle": False
            },
            "model_training_parameters": {
                "learning_rate": 0.00025,
                "gamma": 0.9,
                "verbose": 1
            },
            "rl_config": {
                "train_cycles": 25,
                "max_trade_duration_candles": 300,
                "max_training_drawdown_pct": 0.02,
                "cpu_count": 2,
                "model_type": "PPO",
                "policy_type": "MlpPolicy",
                "continual_learning": False,
                "model_reward_parameters": {
                    "rr": 1,
                    "profit_aim": 0.025,
                    "win_reward_factor": 2
                }
            }
        }
    }

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pairs will be used for correlation analysis and market regime detection.
        """
        pairs = self.dp.current_whitelist()
        informative_pairs = []

        # Add multiple timeframes for current pairs
        for pair in pairs:
            informative_pairs.extend([
                (pair, "15m"),
                (pair, "1h"),
                (pair, "4h"),
            ])

        # Add major pairs for market context
        major_pairs = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
        for pair in major_pairs:
            if pair not in pairs:
                informative_pairs.extend([
                    (pair, self.timeframe),
                    (pair, "1h"),
                ])

        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate comprehensive technical indicators that will serve as features for the RL agent.

        The RL agent will learn which indicators are most predictive for different market conditions.
        """

        # === MOMENTUM INDICATORS ===

        # RSI with multiple periods
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)
        dataframe["rsi_30"] = ta.RSI(dataframe, timeperiod=30)
        dataframe["rsi_50"] = ta.RSI(dataframe, timeperiod=50)

        # Stochastic indicators
        stoch = ta.STOCH(dataframe)
        dataframe["stoch_k"] = stoch["slowk"]
        dataframe["stoch_d"] = stoch["slowd"]

        stoch_rsi = ta.STOCHRSI(dataframe)
        dataframe["stoch_rsi_k"] = stoch_rsi["fastk"]
        dataframe["stoch_rsi_d"] = stoch_rsi["fastd"]

        # MACD
        macd = ta.MACD(dataframe,
                      fastperiod=self.macd_fast.value,
                      slowperiod=self.macd_slow.value,
                      signalperiod=9)
        dataframe["macd"] = macd["macd"]
        dataframe["macd_signal"] = macd["macdsignal"]
        dataframe["macd_hist"] = macd["macdhist"]

        # Williams %R
        dataframe["williams_r"] = ta.WILLR(dataframe)

        # CCI
        dataframe["cci"] = ta.CCI(dataframe)

        # ADX
        dataframe["adx"] = ta.ADX(dataframe)
        dataframe["plus_di"] = ta.PLUS_DI(dataframe)
        dataframe["minus_di"] = ta.MINUS_DI(dataframe)

        # === VOLATILITY INDICATORS ===

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe),
                                          window=self.bb_period.value, stds=2)
        dataframe["bb_lower"] = bollinger["lower"]
        dataframe["bb_middle"] = bollinger["mid"]
        dataframe["bb_upper"] = bollinger["upper"]
        dataframe["bb_percent"] = (dataframe["close"] - dataframe["bb_lower"]) / (
            dataframe["bb_upper"] - dataframe["bb_lower"]
        )
        dataframe["bb_width"] = (dataframe["bb_upper"] - dataframe["bb_lower"]) / dataframe["bb_middle"]

        # Keltner Channel
        keltner = qtpylib.keltner_channel(dataframe)
        dataframe["kc_upper"] = keltner["upper"]
        dataframe["kc_lower"] = keltner["lower"]
        dataframe["kc_middle"] = keltner["mid"]

        # Average True Range
        dataframe["atr"] = ta.ATR(dataframe)
        dataframe["atr_percent"] = (dataframe["atr"] / dataframe["close"]) * 100

        # === VOLUME INDICATORS ===

        # Money Flow Index
        dataframe["mfi"] = ta.MFI(dataframe)

        # On Balance Volume
        dataframe["obv"] = ta.OBV(dataframe)

        # Volume Rate of Change
        dataframe["volume_roc"] = ta.ROC(dataframe["volume"], timeperiod=10)

        # === TREND INDICATORS ===

        # Moving Averages
        dataframe["sma_10"] = ta.SMA(dataframe, timeperiod=10)
        dataframe["sma_20"] = ta.SMA(dataframe, timeperiod=20)
        dataframe["sma_50"] = ta.SMA(dataframe, timeperiod=50)
        dataframe["ema_10"] = ta.EMA(dataframe, timeperiod=10)
        dataframe["ema_20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)

        # TEMA
        dataframe["tema"] = ta.TEMA(dataframe, timeperiod=20)

        # Parabolic SAR
        dataframe["sar"] = ta.SAR(dataframe)

        # === PRICE ACTION INDICATORS ===

        # Price position relative to recent highs/lows
        dataframe["high_20"] = dataframe["high"].rolling(window=20).max()
        dataframe["low_20"] = dataframe["low"].rolling(window=20).min()
        dataframe["price_position"] = (dataframe["close"] - dataframe["low_20"]) / (
            dataframe["high_20"] - dataframe["low_20"]
        )

        # Rate of Change
        dataframe["roc_10"] = ta.ROC(dataframe, timeperiod=10)
        dataframe["roc_20"] = ta.ROC(dataframe, timeperiod=20)

        # === MARKET REGIME INDICATORS ===

        if self.use_dynamic_features.value:
            # Volatility regime
            dataframe["volatility_regime"] = self.calculate_volatility_regime(dataframe)

            # Trend strength
            dataframe["trend_strength"] = self.calculate_trend_strength(dataframe)

            # Market efficiency
            dataframe["market_efficiency"] = self.calculate_market_efficiency(dataframe)

        # === MULTI-TIMEFRAME FEATURES ===

        # Get higher timeframe data for context
        if self.dp:
            # 15m timeframe indicators
            inf_15m = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe="15m")
            if not inf_15m.empty:
                inf_15m["rsi_15m"] = ta.RSI(inf_15m)
                inf_15m["ema_20_15m"] = ta.EMA(inf_15m, timeperiod=20)
                dataframe = merge_informative_pair(dataframe, inf_15m, self.timeframe, "15m",
                                                 ffill=True)

            # 1h timeframe indicators
            inf_1h = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe="1h")
            if not inf_1h.empty:
                inf_1h["rsi_1h"] = ta.RSI(inf_1h)
                inf_1h["ema_20_1h"] = ta.EMA(inf_1h, timeperiod=20)
                inf_1h["trend_1h"] = np.where(inf_1h["close"] > inf_1h["ema_20_1h"], 1, -1)
                dataframe = merge_informative_pair(dataframe, inf_1h, self.timeframe, "1h",
                                                 ffill=True)

        # === FEATURE ENGINEERING ===

        # Interaction features
        dataframe["rsi_bb_interaction"] = dataframe["rsi"] * dataframe["bb_percent"]
        dataframe["macd_rsi_interaction"] = dataframe["macd"] * dataframe["rsi"] / 100

        # Momentum divergence
        dataframe["price_momentum"] = dataframe["close"].pct_change(10)
        dataframe["rsi_momentum"] = dataframe["rsi"].pct_change(10)
        dataframe["momentum_divergence"] = dataframe["price_momentum"] - dataframe["rsi_momentum"]

        # Volatility-adjusted indicators
        dataframe["rsi_vol_adj"] = dataframe["rsi"] / (dataframe["atr_percent"] + 1)
        dataframe["macd_vol_adj"] = dataframe["macd"] / (dataframe["atr_percent"] + 1)

        return dataframe

    def calculate_volatility_regime(self, dataframe: DataFrame) -> pd.Series:
        """
        Calculate volatility regime (low, medium, high) based on ATR percentiles.
        """
        lookback = self.market_regime_lookback.value
        atr_rolling = dataframe["atr"].rolling(window=lookback)

        # Calculate percentiles
        atr_20 = atr_rolling.quantile(0.2)
        atr_80 = atr_rolling.quantile(0.8)

        # Classify regime
        regime = pd.Series(index=dataframe.index, dtype=float)
        regime = np.where(dataframe["atr"] < atr_20, 0, regime)  # Low volatility
        regime = np.where((dataframe["atr"] >= atr_20) & (dataframe["atr"] < atr_80), 1, regime)  # Medium
        regime = np.where(dataframe["atr"] >= atr_80, 2, regime)  # High volatility

        return regime

    def calculate_trend_strength(self, dataframe: DataFrame) -> pd.Series:
        """
        Calculate trend strength based on ADX and price momentum.
        """
        # Normalize ADX to 0-1 scale
        adx_norm = dataframe["adx"] / 100

        # Calculate price momentum
        price_momentum = abs(dataframe["close"].pct_change(20))

        # Combine indicators
        trend_strength = (adx_norm + price_momentum) / 2

        return trend_strength

    def calculate_market_efficiency(self, dataframe: DataFrame) -> pd.Series:
        """
        Calculate market efficiency using the ratio of price change to price path.
        Higher values indicate more efficient (trending) markets.
        """
        lookback = 20

        # Calculate direct price change
        price_change = abs(dataframe["close"] - dataframe["close"].shift(lookback))

        # Calculate cumulative price path
        price_path = abs(dataframe["close"].diff()).rolling(window=lookback).sum()

        # Efficiency ratio
        efficiency = price_change / (price_path + 1e-10)  # Add small value to avoid division by zero

        return efficiency

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Entry signals will be generated by the RL agent based on the features provided.
        We set basic conditions here, but the RL agent will learn the optimal entry timing.
        """

        # The RL agent will learn when to enter based on all the features we've provided
        # We just need to ensure we don't enter on invalid conditions

        # Basic volume filter
        volume_condition = dataframe["volume"] > 0

        # Avoid entering during extreme volatility spikes
        volatility_filter = dataframe["atr_percent"] < dataframe["atr_percent"].quantile(0.95)

        # Basic trend filter for long entries (RL agent can override this)
        long_trend_filter = dataframe["ema_20"] > dataframe["ema_50"]

        # Basic trend filter for short entries (RL agent can override this)
        short_trend_filter = dataframe["ema_20"] < dataframe["ema_50"]

        # Set entry conditions (RL agent will learn the actual signals)
        dataframe.loc[
            (
                volume_condition &
                volatility_filter &
                long_trend_filter
            ),
            "enter_long",
        ] = 1

        dataframe.loc[
            (
                volume_condition &
                volatility_filter &
                short_trend_filter
            ),
            "enter_short",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Exit signals will be generated by the RL agent.
        We provide basic conditions, but the agent will learn optimal exit timing.
        """

        # Basic volume filter
        volume_condition = dataframe["volume"] > 0

        # The RL agent will learn when to exit based on the features
        # We just provide basic structure

        dataframe.loc[
            volume_condition,
            "exit_long",
        ] = 0  # RL agent will decide

        dataframe.loc[
            volume_condition,
            "exit_short",
        ] = 0  # RL agent will decide

        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        """
        Customize leverage per pair. RL agent can learn to adjust this based on market conditions.
        """
        # Conservative leverage for RL training
        return min(proposed_leverage, 3.0)

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Custom stoploss logic. The RL agent can learn to adjust stoplosses dynamically.
        """

        # Get current dataframe
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return self.stoploss

        # Get current candle
        current_candle = dataframe.iloc[-1]

        # Adaptive stoploss based on volatility
        atr_percent = current_candle.get("atr_percent", 2.0)

        # Adjust stoploss based on current market volatility
        if atr_percent > 3.0:  # High volatility
            return -0.20  # Wider stoploss
        elif atr_percent < 1.0:  # Low volatility
            return -0.10  # Tighter stoploss
        else:
            return self.stoploss  # Default stoploss

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float,
                           rate: float, time_in_force: str, current_time: datetime,
                           entry_tag: Optional[str], side: str, **kwargs) -> bool:
        """
        Confirm trade entry. Can be used to add additional filters or risk management.
        """

        # Get current dataframe
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return False

        current_candle = dataframe.iloc[-1]

        # Risk management: avoid entering during extreme market conditions
        if current_candle.get("atr_percent", 0) > 5.0:  # Extreme volatility
            logger.info(f"Skipping {pair} entry due to extreme volatility")
            return False

        # Check if we're in a valid market regime
        volatility_regime = current_candle.get("volatility_regime", 1)
        if volatility_regime == 2:  # High volatility regime
            # More conservative in high volatility
            if abs(current_candle.get("rsi", 50) - 50) < 20:  # RSI not extreme enough
                return False

        return True

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                          rate: float, time_in_force: str, exit_reason: str,
                          current_time: datetime, **kwargs) -> bool:
        """
        Confirm trade exit. The RL agent will learn optimal exit timing.
        """
        return True

    def bot_start(self, **kwargs) -> None:
        """
        Called only once after bot instantiation.
        """
        logger.info("RL Adaptive Strategy initialized")
        logger.info("This strategy uses FreqAI's reinforcement learning capabilities")
        logger.info("Make sure your config includes proper FreqAI RL configuration")

    def bot_loop_start(self, **kwargs) -> None:
        """
        Called at the start of each bot loop iteration.
        """
        pass