from dataclasses import dataclass, field
import numpy as np
from typing import Any

@dataclass(frozen=True)
class UnderlyingState:
    """
    A generic current state representation of a derivative's underlying.

    Args:
        time: Time since inception, in years. Must be non-negative.
        spot: Underlying price (scalar or array of paths for MC simulation). 
        Must be non-negative.
        extra_params: Optional additional model parameters (e.g. the
            instantaneous variance under the Heston model. The Heston
            model is not yet implemented, but it is intended as an expansion 
            for future developments).

    Raises:
        ValueError: if `time` or `spot` are negative.
    """
    time: float
    spot: float | np.ndarray
    extra_params: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if np.any(np.asarray(self.time) < 0):
            raise ValueError(f"time must be non-negative, got {self.time}")
        if np.any(np.asarray(self.spot) < 0):
            raise ValueError(f"spot must be non-negative, got {self.spot}")
        
    def get_extra_param(self, name: str) -> float:
        """
        Returns the value of a state extra parameter, if available. 
        
        Args:
            name: key of parameter in the extra state dictionary.

        Raises:
            KeyError: if `name` is not defined within the state.
        """
        try:
            return self.extra_params[name]
        except KeyError:
            raise KeyError(
                f"UnderlyingState instance does not contain param '{name}'"
                f"Available params are: {list(self.extra_params.keys())}"
            )


@dataclass(frozen=True)
class PathDependentState:
    """
    Additional path-dependent state required by a derivative (e.g. running
    average for an Asian option, knock-out flag for a barrier).

    Args:
        params: Mapping from a `PathFeatureTracker.name` to its current value
            (scalar, or array across paths).
    """
    params: dict[str, Any] = field(default_factory=dict)

    def get_param(self, name: str) -> float:
        """
        Returns the value of a contract state parameter if available.

        Args:
            name: key of parameter in the contract state dictionary.
 
        Raises:
            KeyError: if `name` is not defined within the state.
        """
        try:
            return self.params[name]
        except KeyError:
            raise KeyError(
                f"PathDependentState instance does not contain param '{name}'. "
                f"Available params are: {list(self.params.keys())}. "
            )

@dataclass(frozen=True)
class PricingState:
    """
    Complete state seen by a derivative contract: an `UnderlyingState` plus its
        `PathDependentState`. This is what `Payoff.__call__` and
        `ExerciseRights.is_exercisable` are evaluated on.
    """
    underlying_state: UnderlyingState
    path_state: PathDependentState

    @property
    def time(self):
        return self.underlying_state.time

    @property
    def spot(self):
        return self.underlying_state.spot
