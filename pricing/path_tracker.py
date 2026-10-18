from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, ClassVar
import numpy as np
from pricing.state import PathDependentState, PricingState, UnderlyingState


class PathFeatureTracker(ABC):
    """
    Computes and evolves one piece of path-dependent contract state (e.g. a
    running average, a knock-out flag) alongside the underlying's own path.
 
    An `Option` declares which augmenters it needs via `required_states`;
    a pricing engine is responsible for calling `initialize` once at t=0
    and `update` it accordingly, storing the result in `PathDependentState.params` under `name`.
    """

    ### TODO: UNDERSTAND BETTER THE _REGISTRY CODE
    name: str
    _registry: ClassVar[dict[str, type[PathFeatureTracker]]] = {}
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = cls.__dict__.get("name")
        if name is None:  # abstract intermediate class
            return
        if not isinstance(name, str):
            raise TypeError(f"{cls.__name__}.name must be a string")
        existing = PathFeatureTracker._registry.get(name)
        if existing is not None and existing.__qualname__ != cls.__qualname__:
            raise ValueError(
                f"Tracker name '{name}' already registered by {existing.__name__}"
            )
        PathFeatureTracker._registry[name] = cls

    @classmethod
    def registered_names(cls) -> frozenset[str]:
        return frozenset(PathFeatureTracker._registry)

    @abstractmethod
    def initialize(self, initial_underlying_state: UnderlyingState) -> Any:
        """Value of the tracked feature at t=0."""

    @abstractmethod
    def update(self, previous_value: Any, previous_state: UnderlyingState,
                current_state: UnderlyingState, step: int) -> Any:
        """
        New value after the underlying moved from `previous_state` to
        `current_state`.

        Args:
            previous_value: this tracker's value at `previous_state`.
            previous_state: underlying before the step.
            current_state: underlying after the step.
            step: zero-based index of this step (0 for the first step after
                initialization). Every engine must follow this convention.
        """


class RunningAverage(PathFeatureTracker):
    """
    Arithmetic running average of the spot, sampled at t=0 and after each
    simulation step (so after step k it averages k+2 samples).
    """
    name = 'running_average'

    def initialize(self, initial_underlying_state: UnderlyingState) -> Any:
            return initial_underlying_state.spot
    
    def update(self, previous_value, previous_state, current_state, step: int):
        n_previous = step + 1  # samples already in the average
        return (n_previous * previous_value + current_state.spot) / (n_previous + 1)


_VALID_DIRECTIONS = frozenset({"up", "down"})
def validate_barrier_direction(direction: str) -> None:
    if direction not in _VALID_DIRECTIONS:
        raise ValueError(
            f"Accepted direction values are 'up' or 'down', got '{direction}'."
        )

class KnockOutStatus(PathFeatureTracker):
    """
    Absorbing "has the barrier been hit" flag: once True it stays True.
    Monitoring is discrete (on simulation / tree dates only).

    Args:
        barrier_level: strictly positive barrier.
        direction: 'up' (hit when spot >= barrier) or 'down' (spot <= barrier).
    """
    name = "knock_out"

    def __init__(self, barrier_level: float, direction: str):
        if barrier_level <= 0:
            raise ValueError(f"Barrier level must be positive, got {barrier_level}.")
        validate_barrier_direction(direction)
        self.barrier_level = barrier_level
        self.direction = direction
 
    def _breached(self, current_state: UnderlyingState) -> bool:
        if self.direction == 'up':
            return self.barrier_level < current_state.spot
        return self.barrier_level > current_state.spot
 
    def initialize(self, initial_underlying_state: UnderlyingState) -> bool:
        return self._breached(initial_underlying_state)

    def update(self, previous_value, previous_state, current_state, step: int):
        return np.logical_or(previous_value, self._breached(current_state))


def collect_trackers_by_name(names: set[str], 
                             tracker_params: dict[str, dict[str, Any]] | None = None
                             ) -> PathFeatureTracker | None:
    """Instantiate the registered trackers `names`, using `tracker_params[name]`
        as constructor kwargs (empty if absent)."""
    tracker_params = tracker_params or {}
    names = set(names)
    unknown = names - PathFeatureTracker.registered_names()
    if unknown:
        raise ValueError(
            f"Unknown trackers {sorted(unknown)}; "
            f"registered: {sorted(PathFeatureTracker.registered_names())}"
        )
    return [
        PathFeatureTracker._registry[n](**tracker_params.get(n, {}))
        for n in sorted(names)
    ]

def build_pricing_path(underlying_history: Sequence[UnderlyingState],
                       trackers: Sequence[PathFeatureTracker]) -> list[PricingState]:
    """
    Apply `trackers` along an underlying history and return one PricingState per
    time point for simulation-based engines.
    """
    path_state = PathDependentState(
        params={t.name: t.initialize(underlying_history[0]) for t in trackers}
    )
    states = [PricingState(underlying_history[0], path_state)]
    for step in range(len(underlying_history) - 1):
        previous, current = underlying_history[step], underlying_history[step + 1]
        path_state = PathDependentState(
            params={
                t.name: t.update(
                    previous_value=path_state.get_param(t.name),
                    previous_state=previous,
                    current_state=current,
                    step=step,
                )
                for t in trackers
            }
        )
        states.append(PricingState(current, path_state))
    return states