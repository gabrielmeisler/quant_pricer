from abc import ABC, abstractmethod
import numpy as np
from pricing.path_tracker import validate_barrier_direction
from pricing.state import PricingState

class Payoff(ABC):
    """A payoff function evaluated on a `PricingState` (scalar or array of paths)."""
    
    @property
    def required_path_trackers(self) -> frozenset:
        """Names of the `PathFeatureTracker`s this payoff reads."""
        return frozenset()

    @property
    def tracker_configs(self) -> dict:
        """
        Constructor kwargs per tracker name; override when trackers need
        parameters, e.g. {'knock_out': {'barrier_level': 100, 'direction': 'up'}}.
        """
        return {}
    
    @abstractmethod
    def __call__(self, state: PricingState) -> float | np.ndarray:
        pass

class VanillaPayoff(Payoff):
    """
    Standard European-style call/put intrinsic value: max(0, spot - strike)
    for a call, max(0, strike - spot) for a put.

    Args:
        strike: Strike price. Must be strictly positive.
        is_call: True for a call, False for a put.

    Raises:
        ValueError: if `strike` is not positive.
    """
    def __init__(self, strike: float, is_call: bool):
        if strike <= 0:
            raise ValueError(f"strike must be positive, got {strike}")
        self.strike = strike
        self.is_call = is_call

    def __call__(self, state: PricingState) -> float | np.ndarray:
        if self.is_call:
            return np.maximum(0.0, state.spot - self.strike)
        return np.maximum(0.0, self.strike - state.spot)

class DigitalPayoff(Payoff):
    """
    Cash-or-nothing digital. Pays `cash_payout` if spot is on the 
    in-the-money at evaluation, otherwise 0.
 
    Args:
        strike: Strike (trigger) price. Must be strictly positive.
        is_call: True pays out when spot >= strike, False when spot <= strike.
        cash_payout: Amount paid out when in the money. Must be non-negative.
 
    Raises:
        ValueError: if `strike` is not positive or `cash_payout` is negative.
    """
    def __init__(self, strike: float, is_call: bool, cash_payout: float = 1.0):
        if strike <= 0:
            raise ValueError(f"strike must be positive, got {strike}")
        if cash_payout < 0:
            raise ValueError(f"cash_payout must be non-negative, got {cash_payout}")
        self.strike = strike
        self.is_call = bool(is_call)
        self.cash_payout = cash_payout

    def __call__(self, state: PricingState) -> float | np.ndarray:
        if self.is_call:
            in_the_money = state.spot >= self.strike
        else:
            in_the_money = state.spot <= self.strike
        return np.where(in_the_money, self.cash_payout, 0.0)

class AsianPayoff(Payoff):
    """
    Average-price call/put. Like a `VanillaPayoff` but struck against the
    running average of spot rather than the spot itself at evaluation time.
 
    Args:
        strike: Strike price. Must be strictly positive.
        is_call: True for a call, False for a put.
 
    Raises:
        ValueError: if `strike` is not positive.
    """
    def __init__(self, strike: float, is_call: bool):
        if strike <= 0:
            raise ValueError(f"strike must be positive, got {strike}")
        self.strike = strike
        self.is_call = bool(is_call)
 
    @property
    def required_path_trackers(self) -> frozenset:
        return frozenset({'running_average'})

    @property
    def tracker_configs(self) -> dict:
        return {
            'running_average': {}
        }
 
    def __call__(self, state: PricingState) -> float | np.ndarray:
        avg_price = state.path_state.get_param('running_average')
        if self.is_call:
            return np.maximum(0.0, avg_price - self.strike)
        return np.maximum(0.0, self.strike - avg_price)

class BarrierPayoff(Payoff):
    """
    Knock-out vanilla option. The vanilla payoff, or 0 if
    the barrier has been hit on any monitoring date. 
    
    Args:
        strike: Strike price. Must be strictly positive.
        is_call: True for a call, False for a put.
        barrier_level: Barrier for knock out.
        barrier_direction: Direction for barrier mechanism definition.
     
    Raises:
        ValueError: if `strike` or `barrier_level` is not positive.
    """
    
    def __init__(self, strike: float, is_call: bool, 
                 barrier_level: float, barrier_direction: str):
        if strike <= 0:
            raise ValueError(f"strike must be positive, got {strike}")
        if barrier_level <= 0:
            raise ValueError(f"barrier_level must be positive, got {barrier_level}")
        validate_barrier_direction(barrier_direction)  

        self.strike = strike
        self.is_call = is_call
        self.barrier_level = barrier_level
        self.barrier_direction = barrier_direction
        self._vanilla = VanillaPayoff(strike, is_call)

    @property
    def required_path_trackers(self) -> frozenset:
        return frozenset({'knock_out'})
    
    @property
    def tracker_configs(self) -> dict:
        return {
            'knock_out': {
                'barrier_level': self.barrier_level,
                'direction': self.barrier_direction
            }
        }
    
    def __call__(self, state: PricingState) -> float | np.ndarray:
        is_knocked = state.path_state.get_param("knock_out")
        intrinsic = self._vanilla(state)
        if np.isscalar(is_knocked):
            return 0.0 if is_knocked else intrinsic
        return np.where(is_knocked, 0.0, intrinsic)