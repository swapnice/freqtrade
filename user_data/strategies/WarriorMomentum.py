# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Optional, Union

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,  # @informative decorator
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

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
from technical import qtpylib


# This class is a sample. Feel free to customize it.
class WarriorMomentum(IStrategy):
    """
    Selecting cryptos that are moving 20-30% in a day.
    Criteria #1: Strong Daily Charts (above the Moving Averages and with no nearby resistance).
    Criteria #2: High Relative Volume of at least 2x above average. (This compares the current volume for today to the average volume for this time of day. These all refer to the standard volume numbers, which are reset every night at midnight.)

    With the Bull Flag Pattern, my entry is the first candle to make a new high after the breakout.

    Entry Rules:
    Entry Criteria #1: Momentum Day Trading Chart Pattern (Bull Flag or Flat Top Breakout)
    Entry Criteria #2: You have a tight stop that supports a 2:1 profit loss ratio
    Entry Criteria #3: You have high relative volume (2x or higher) and ideally associated with a catalyst. Heavier volume means more people are watching.

    Exit Rules:
    Exit Indicator #1: I will sell 1/2 when I hit my first profit target. If I'm risking $100 to make $200, once I'm up $200 I'll sell 1/2. I then adjust my stop to my entry price on the balance of my position
    Exit Indicator #2: If I haven't already sold 1/2, the first candle to close red is an exit indicator. If I've already sold 1/2, I'll hold through red candles as long as my breakeven stop doesn't hit.
    Exit Indicator #3: Extension bar forces me to begin locking in my profits before the inevitable reversal begins. An extension bar is a candle that spikes up and instantly put my up $200,400 or more. When I'm lucky enough to have a crypto spike up while I'm holding, I sell into the spike.


    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_entry_trend, populate_exit_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """

    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "0": 0.2,  # Increased ROI target to be more selective
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.1  # Loosened stop to allow for more volatility

    # Trailing stoploss
    trailing_stop = False
    # trailing_stop_positive = 0.02
    # trailing_stop_positive_offset = 0.03
    # trailing_only_offset_is_reached = True

    # Optimal timeframe for the strategy.
    timeframe = "1m"

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Hyperoptable parameters - Made more lenient to generate trades
    volume_multiplier = DecimalParameter(1.2, 2.5, default=1.5, space="buy")
    rsi_buy_min = IntParameter(30, 50, default=40, space="buy")
    rsi_buy_max = IntParameter(65, 85, default=80, space="buy")
    rsi_sell = IntParameter(75, 95, default=85, space="sell")
    momentum_threshold = DecimalParameter(0.005, 0.02, default=0.01, space="buy")

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    # Optional order type mapping.
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    # Optional order time in force.
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    plot_config = {
        "main_plot": {
            "tema": {},
            "sar": {"color": "white"},
            "ema9": {"color": "blue"},
            "ema21": {"color": "orange"},
        },
        "subplots": {
            "MACD": {
                "macd": {"color": "blue"},
                "macdsignal": {"color": "orange"},
            },
            "RSI": {
                "rsi": {"color": "red"},
            },
            "Volume": {
                "volume": {"color": "green"},
                "avg_volume": {"color": "red"},
            },
        },
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """

        # Momentum Indicators
        # ------------------------------------

        # ADX
        dataframe["adx"] = ta.ADX(dataframe)

        # RSI
        dataframe["rsi"] = ta.RSI(dataframe)

        # Stochastic Fast
        stoch_fast = ta.STOCHF(dataframe)
        dataframe["fastd"] = stoch_fast["fastd"]
        dataframe["fastk"] = stoch_fast["fastk"]

        # MACD
        macd = ta.MACD(dataframe)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        dataframe["macdhist"] = macd["macdhist"]

        # MFI
        dataframe["mfi"] = ta.MFI(dataframe)

        # Overlap Studies
        # ------------------------------------

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_middleband"] = bollinger["mid"]
        dataframe["bb_upperband"] = bollinger["upper"]
        dataframe["bb_percent"] = (dataframe["close"] - dataframe["bb_lowerband"]) / (
            dataframe["bb_upperband"] - dataframe["bb_lowerband"]
        )
        dataframe["bb_width"] = (dataframe["bb_upperband"] - dataframe["bb_lowerband"]) / dataframe[
            "bb_middleband"
        ]

        # EMA - Exponential Moving Average
        dataframe['ema9'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema21'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)

        # Parabolic SAR
        dataframe["sar"] = ta.SAR(dataframe)

        # TEMA - Triple Exponential Moving Average
        dataframe["tema"] = ta.TEMA(dataframe, timeperiod=9)

        # Volume indicators
        # ------------------------------------
        # Average volume
        dataframe["avg_volume"] = dataframe["volume"].rolling(20).mean()
        dataframe["volume_ratio"] = dataframe["volume"] / dataframe["avg_volume"]

        # Recent high for breakout detection
        dataframe["recent_high"] = dataframe["high"].rolling(20).max().shift(1)
        dataframe["recent_high_5"] = dataframe["high"].rolling(5).max().shift(1)
        dataframe["recent_high_10"] = dataframe["high"].rolling(10).max().shift(1)

        # Bull flag pattern detection - made more lenient
        dataframe["consolidation"] = (
            (dataframe["high"].rolling(10).max() - dataframe["low"].rolling(10).min()) /
            dataframe["close"]
        ) < 0.05  # Loosened consolidation requirement

        # Momentum indicators
        dataframe["price_change_pct"] = (dataframe["close"] - dataframe["open"]) / dataframe["open"]
        dataframe["momentum_5"] = (dataframe["close"] - dataframe["close"].shift(5)) / dataframe["close"].shift(5)

        # Cycle Indicator
        # ------------------------------------
        # Hilbert Transform Indicator - SineWave
        hilbert = ta.HT_SINE(dataframe)
        dataframe["htsine"] = hilbert["sine"]
        dataframe["htleadsine"] = hilbert["leadsine"]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """
        dataframe.loc[
            (
                # Price filter: ignore cryptos above $0.90 USDT
                (dataframe["close"] <= 0.90)
                &
                # Reduced volume requirement
                (dataframe["volume_ratio"] >= self.volume_multiplier.value)
                &
                # More lenient breakout patterns
                (
                    # Bull flag: consolidation followed by breakout
                    (
                        (dataframe["consolidation"].shift(1) == True) &
                        (dataframe["high"] > dataframe["recent_high_10"])
                    )
                    |
                    # Flat top breakout: new high with volume
                    (
                        (dataframe["high"] > dataframe["recent_high"]) &
                        (dataframe["volume_ratio"] >= 1.5)  # Reduced from 2.5
                    )
                    |
                    # Simple momentum breakout
                    (
                        (dataframe["close"] > dataframe["ema9"]) &
                        (dataframe["momentum_5"] > 0.01)
                    )
                )
                &
                # Reduced momentum requirement
                (dataframe["price_change_pct"] >= self.momentum_threshold.value)
                &
                # Wider RSI range
                (dataframe["rsi"] >= self.rsi_buy_min.value)
                & (dataframe["rsi"] <= self.rsi_buy_max.value)
                &
                # More lenient MACD
                (dataframe["macd"] > dataframe["macdsignal"] * 0.95)  # Allow slight negative
                &
                # Price above key moving averages - made more lenient
                (
                    (dataframe["close"] > dataframe["ema9"]) |
                    (dataframe["close"] > dataframe["ema21"])
                )
                &
                # Reduced ADX requirement
                (dataframe["adx"] > 20)  # Reduced from 25
                &
                # Volume confirmation
                (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with exit columns populated
        """
        dataframe.loc[
            (
                # Multiple exit conditions - made more conservative to hold longer
                (
                    # RSI very overbought
                    (dataframe["rsi"] > self.rsi_sell.value)
                    &
                    # MACD bearish crossover with confirmation
                    (
                        (dataframe["macd"] < dataframe["macdsignal"]) &
                        (dataframe["macd"].shift(1) >= dataframe["macdsignal"].shift(1)) &
                        (dataframe["macd"] < dataframe["macdsignal"] * 1.02)
                    )
                    &
                    # Price below both key support levels
                    (
                        (dataframe["close"] < dataframe["ema9"]) &
                        (dataframe["close"] < dataframe["ema21"])
                    )
                )
                & (dataframe["volume"] > 0)
            ),
            "exit_long",
        ] = 1

        return dataframe

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Optional[str]:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) < 2:
            return None

        # last_candle = dataframe.iloc[-1].squeeze()
        # previous_candle = dataframe.iloc[-2].squeeze()

        # Exit on first red candle only if profit is very minimal
        # if (previous_candle["close"] < previous_candle["open"] and
        #     current_profit < 0.005):  # Less than 0.5% profit
        #     return "first_red_candle"

        # Exit on extension bar (large spike) - increased threshold
        # candle_change = (last_candle["close"] - last_candle["open"]) / last_candle["open"]
        # if candle_change > 0.08:  # 8% spike in single candle
        #     return "extension_bar_spike"

        # Exit on severe volume exhaustion with strong negative momentum
        # if (last_candle["volume_ratio"] < 0.2 and
        #     last_candle["momentum_5"] < -0.02):
        #     return "volume_exhaustion"

        return None
