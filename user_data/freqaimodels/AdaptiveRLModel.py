import logging
import numpy as np
import pandas as pd
from typing import Any, Dict
from pathlib import Path
from datetime import datetime

import torch as th
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.RL.Base5ActionRLEnv import Actions, Base5ActionRLEnv, Positions
from freqtrade.freqai.RL.BaseEnvironment import BaseEnvironment
from freqtrade.freqai.prediction_models.ReinforcementLearner import ReinforcementLearner


logger = logging.getLogger(__name__)


class AdaptiveRLModel(ReinforcementLearner):
    """
    Adaptive Reinforcement Learning Model for Freqtrade

    This model extends the base ReinforcementLearner with:
    - Adaptive reward functions based on market conditions
    - Multi-objective optimization (profit, drawdown, trade frequency)
    - Dynamic feature importance tracking
    - Market regime-aware training
    - Advanced risk management integration

    Usage:
    Set this as your freqaimodel in the config:
    "freqaimodel": "AdaptiveRLModel"
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reward_history = []
        self.market_regime_history = []
        self.feature_importance_history = []

    def fit(self, data_dictionary: Dict[str, Any], dk: FreqaiDataKitchen, **kwargs):
        """
        Enhanced fit method with adaptive training based on market conditions.
        """

        # Analyze market conditions to adjust training parameters
        market_conditions = self.analyze_market_conditions(data_dictionary)

        # Adjust training cycles based on market volatility
        base_cycles = self.freqai_info["rl_config"]["train_cycles"]
        if market_conditions["volatility_regime"] == "high":
            # More training in volatile markets
            adjusted_cycles = int(base_cycles * 1.5)
        elif market_conditions["volatility_regime"] == "low":
            # Less training in stable markets
            adjusted_cycles = int(base_cycles * 0.8)
        else:
            adjusted_cycles = base_cycles

        logger.info(f"Adjusting training cycles from {base_cycles} to {adjusted_cycles} "
                   f"based on {market_conditions['volatility_regime']} volatility regime")

        # Update training parameters
        original_cycles = self.freqai_info["rl_config"]["train_cycles"]
        self.freqai_info["rl_config"]["train_cycles"] = adjusted_cycles

        # Call parent fit method
        try:
            model = super().fit(data_dictionary, dk, **kwargs)
        finally:
            # Restore original cycles
            self.freqai_info["rl_config"]["train_cycles"] = original_cycles

        # Log training summary
        self.log_training_summary(market_conditions)

        return model

    def analyze_market_conditions(self, data_dictionary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market conditions to adapt training parameters.
        """
        train_df = data_dictionary["train_features"]

        # Calculate volatility metrics
        if "atr_percent" in train_df.columns:
            volatility = train_df["atr_percent"].mean()
            volatility_std = train_df["atr_percent"].std()
        else:
            # Fallback to price volatility
            volatility = train_df["close"].pct_change().std() * 100
            volatility_std = 0

        # Determine volatility regime
        if volatility > 3.0:
            volatility_regime = "high"
        elif volatility < 1.0:
            volatility_regime = "low"
        else:
            volatility_regime = "medium"

        # Calculate trend strength
        if "adx" in train_df.columns:
            trend_strength = train_df["adx"].mean()
        else:
            trend_strength = 25  # Default

        # Calculate market efficiency
        if "market_efficiency" in train_df.columns:
            market_efficiency = train_df["market_efficiency"].mean()
        else:
            market_efficiency = 0.5  # Default

        return {
            "volatility": volatility,
            "volatility_std": volatility_std,
            "volatility_regime": volatility_regime,
            "trend_strength": trend_strength,
            "market_efficiency": market_efficiency,
            "sample_size": len(train_df)
        }

    def log_training_summary(self, market_conditions: Dict[str, Any]):
        """
        Log comprehensive training summary.
        """
        logger.info("=== Training Summary ===")
        logger.info(f"Market Volatility: {market_conditions['volatility']:.2f}%")
        logger.info(f"Volatility Regime: {market_conditions['volatility_regime']}")
        logger.info(f"Trend Strength: {market_conditions['trend_strength']:.2f}")
        logger.info(f"Market Efficiency: {market_conditions['market_efficiency']:.3f}")
        logger.info(f"Training Samples: {market_conditions['sample_size']}")
        logger.info("========================")

    class MyRLEnv(Base5ActionRLEnv):
        """
        Custom RL Environment with adaptive reward functions and advanced features.
        """

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.reward_components = {
                "profit": 0.0,
                "risk": 0.0,
                "efficiency": 0.0,
                "consistency": 0.0
            }
            self.trade_metrics = {
                "win_rate": 0.0,
                "avg_trade_duration": 0.0,
                "max_consecutive_losses": 0,
                "current_consecutive_losses": 0
            }

        def calculate_reward(self, action: int) -> float:
            """
            Advanced reward function with multiple objectives:
            1. Profit maximization
            2. Risk minimization
            3. Trading efficiency
            4. Consistency rewards
            """

            # Base validation
            if not self._is_valid(action):
                self.tensorboard_log("invalid_action", category="actions")
                return -5.0  # Strong penalty for invalid actions

            # Get current market state
            current_features = self.get_current_features()
            market_regime = self.get_market_regime(current_features)

            # Calculate reward components
            profit_reward = self.calculate_profit_reward(action)
            risk_reward = self.calculate_risk_reward(action)
            efficiency_reward = self.calculate_efficiency_reward(action)
            consistency_reward = self.calculate_consistency_reward(action)

            # Adaptive weighting based on market regime
            weights = self.get_adaptive_weights(market_regime)

            # Combined reward
            total_reward = (
                weights["profit"] * profit_reward +
                weights["risk"] * risk_reward +
                weights["efficiency"] * efficiency_reward +
                weights["consistency"] * consistency_reward
            )

            # Store reward components for analysis
            self.reward_components = {
                "profit": profit_reward,
                "risk": risk_reward,
                "efficiency": efficiency_reward,
                "consistency": consistency_reward
            }

            # Log to tensorboard
            self.tensorboard_log("total_reward", total_reward, category="rewards")
            self.tensorboard_log("profit_reward", profit_reward, category="rewards")
            self.tensorboard_log("risk_reward", risk_reward, category="rewards")
            self.tensorboard_log("efficiency_reward", efficiency_reward, category="rewards")
            self.tensorboard_log("consistency_reward", consistency_reward, category="rewards")

            return total_reward

        def calculate_profit_reward(self, action: int) -> float:
            """
            Calculate profit-based reward component.
            """
            pnl = self.get_unrealized_profit()
            factor = 100.0

            # Entry rewards
            if action in (Actions.Long_enter.value, Actions.Short_enter.value) and self._position == Positions.Neutral:
                # Use technical indicators to assess entry quality
                rsi = self.get_feature_value("rsi", default=50)
                bb_percent = self.get_feature_value("bb_percent", default=0.5)

                # Reward contrarian entries
                if action == Actions.Long_enter.value:
                    if rsi < 30 and bb_percent < 0.2:  # Oversold conditions
                        return 30.0
                    elif rsi < 40 and bb_percent < 0.3:
                        return 20.0
                    else:
                        return 10.0

                elif action == Actions.Short_enter.value:
                    if rsi > 70 and bb_percent > 0.8:  # Overbought conditions
                        return 30.0
                    elif rsi > 60 and bb_percent > 0.7:
                        return 20.0
                    else:
                        return 10.0

            # Exit rewards
            if action in (Actions.Long_exit.value, Actions.Short_exit.value):
                if self._position in (Positions.Long, Positions.Short):
                    # Reward profitable exits
                    if pnl > 0:
                        profit_factor = min(pnl * 1000, 100)  # Cap at 100
                        return profit_factor
                    else:
                        # Small penalty for losses, but not too harsh
                        return pnl * 50

            # Neutral action rewards/penalties
            if action == Actions.Neutral.value:
                if self._position == Positions.Neutral:
                    # Small penalty for inaction
                    return -0.5
                else:
                    # Reward holding profitable positions
                    if pnl > 0:
                        return min(pnl * 10, 5)
                    else:
                        # Penalty for holding losing positions
                        return max(pnl * 20, -5)

            return 0.0

        def calculate_risk_reward(self, action: int) -> float:
            """
            Calculate risk-based reward component.
            """
            current_drawdown = 1 - self._total_profit
            max_allowed_drawdown = 1 - self.rl_config.get("max_training_drawdown_pct", 0.8)

            # Penalty for high drawdown
            if current_drawdown > max_allowed_drawdown * 0.5:
                risk_penalty = -20 * (current_drawdown / max_allowed_drawdown)
            else:
                risk_penalty = 0

            # Reward risk management
            if action in (Actions.Long_exit.value, Actions.Short_exit.value):
                pnl = self.get_unrealized_profit()
                if pnl < -0.02:  # Cutting losses
                    return 10.0

            # Penalty for excessive trade duration
            if self._last_trade_tick is not None:
                trade_duration = self._current_tick - self._last_trade_tick
                max_duration = self.rl_config.get("max_trade_duration_candles", 300)

                if trade_duration > max_duration:
                    duration_penalty = -1 * (trade_duration / max_duration)
                    return duration_penalty + risk_penalty

            return risk_penalty

        def calculate_efficiency_reward(self, action: int) -> float:
            """
            Calculate trading efficiency reward.
            """
            # Reward quick profitable trades
            if action in (Actions.Long_exit.value, Actions.Short_exit.value):
                if self._last_trade_tick is not None:
                    trade_duration = self._current_tick - self._last_trade_tick
                    pnl = self.get_unrealized_profit()

                    if pnl > 0 and trade_duration < 50:  # Quick profit
                        efficiency_bonus = 5.0 * (50 - trade_duration) / 50
                        return efficiency_bonus

            # Penalty for overtrading
            recent_trades = len([t for t in self.trade_history[-10:] if t])
            if recent_trades > 5:
                return -2.0

            return 0.0

        def calculate_consistency_reward(self, action: int) -> float:
            """
            Calculate consistency-based reward.
            """
            # Reward maintaining positive equity curve
            if len(self.close_trade_profit) > 5:
                recent_profits = self.close_trade_profit[-5:]
                win_rate = sum(1 for p in recent_profits if p > 0) / len(recent_profits)

                if win_rate > 0.6:
                    return 5.0
                elif win_rate < 0.3:
                    return -3.0

            return 0.0

        def get_current_features(self) -> Dict[str, float]:
            """
            Get current feature values for decision making.
            """
            if self._current_tick >= len(self.signal_features):
                return {}

            current_row = self.signal_features.iloc[self._current_tick]
            return current_row.to_dict()

        def get_feature_value(self, feature_name: str, default: float = 0.0) -> float:
            """
            Safely get feature value with fallback.
            """
            try:
                if hasattr(self, 'raw_features') and feature_name in self.raw_features.columns:
                    return self.raw_features[feature_name].iloc[self._current_tick]
                elif feature_name in self.signal_features.columns:
                    return self.signal_features[feature_name].iloc[self._current_tick]
                else:
                    return default
            except (IndexError, KeyError):
                return default

        def get_market_regime(self, features: Dict[str, float]) -> str:
            """
            Determine current market regime.
            """
            volatility = features.get("atr_percent", 2.0)
            trend_strength = features.get("adx", 25.0)

            if volatility > 3.0:
                return "high_volatility"
            elif volatility < 1.0:
                return "low_volatility"
            elif trend_strength > 40:
                return "trending"
            elif trend_strength < 20:
                return "ranging"
            else:
                return "normal"

        def get_adaptive_weights(self, market_regime: str) -> Dict[str, float]:
            """
            Get adaptive weights based on market regime.
            """
            if market_regime == "high_volatility":
                return {
                    "profit": 0.4,
                    "risk": 0.4,
                    "efficiency": 0.1,
                    "consistency": 0.1
                }
            elif market_regime == "low_volatility":
                return {
                    "profit": 0.6,
                    "risk": 0.2,
                    "efficiency": 0.1,
                    "consistency": 0.1
                }
            elif market_regime == "trending":
                return {
                    "profit": 0.5,
                    "risk": 0.2,
                    "efficiency": 0.2,
                    "consistency": 0.1
                }
            elif market_regime == "ranging":
                return {
                    "profit": 0.3,
                    "risk": 0.3,
                    "efficiency": 0.3,
                    "consistency": 0.1
                }
            else:  # normal
                return {
                    "profit": 0.4,
                    "risk": 0.3,
                    "efficiency": 0.2,
                    "consistency": 0.1
                }

        def reset(self, seed=None):
            """
            Enhanced reset with metrics tracking.
            """
            # Update trade metrics before reset
            if len(self.close_trade_profit) > 0:
                self.trade_metrics["win_rate"] = sum(1 for p in self.close_trade_profit if p > 0) / len(self.close_trade_profit)

            if len(self.trade_history) > 0:
                durations = [t.get("duration", 0) for t in self.trade_history if t]
                if durations:
                    self.trade_metrics["avg_trade_duration"] = sum(durations) / len(durations)

            return super().reset(seed)

        def step(self, action):
            """
            Enhanced step with additional logging and metrics.
            """
            # Store pre-step state
            pre_step_profit = self._total_profit
            pre_step_position = self._position

            # Execute step
            observation, reward, done, truncated, info = super().step(action)

            # Additional logging
            info.update({
                "market_regime": self.get_market_regime(self.get_current_features()),
                "reward_components": self.reward_components.copy(),
                "trade_metrics": self.trade_metrics.copy()
            })

            return observation, reward, done, truncated, info