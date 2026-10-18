from abc import ABC, abstractmethod
from typing import Sequence
import numpy as np

from pricing.state import PricingState

_TIME_TOLERANCE = 1e-10
_GRID_TOLERANCE = 1e-8

def _same_time(t: float, target: float) -> bool:
    # rtol=0: np.isclose's default rtol (1e-5) would dominate atol.
    return bool(np.isclose(t, target, rtol=0.0, atol=_TIME_TOLERANCE))

class ExerciseRights(ABC):
    """Determines when a contract may be exercised."""

    def __init__(self, maturity: float):
        if maturity <= 0:
            raise ValueError(f"maturity must be positive, got {maturity}")
        self.maturity = maturity

    @abstractmethod
    def is_exercisable(self, state: PricingState) -> bool:
        """True if the contract may be exercised at `state`."""

    def check_grid(self, n_steps: int) -> None:
        """Raise if the rights cannot be represented on a uniform grid of
        `n_steps` steps over [0, maturity]. Default: always representable."""

    
class EuropeanExercise(ExerciseRights):
    """Exercisable only at maturity."""
        
    def is_exercisable(self, state: PricingState) -> bool:
        """True at maturity, False everywhere else."""
        return np.isclose(state.time, self.maturity, atol=_TIME_TOLERANCE, rtol=0)

class AmericanExercise(ExerciseRights):
    """Exercisable at any time."""

    def is_exercisable(self, state: PricingState) -> bool:
        """Always True."""
        return True

class BermudanExercise(ExerciseRights):
    """
    Exercisable at a discrete set of times; maturity is always included.

    Args:
        maturity: final exercise time (> 0).
        exercise_times: non-empty, each in (0, maturity].

    Raises:
        ValueError: on empty / non-positive / beyond-maturity times.
    """
    def __init__(self, maturity: float, exercise_times: Sequence[float]):
        super().__init__(maturity)
        times = list(exercise_times)
        if not times:
            raise ValueError("exercise_times must be non-empty")
        if any(t <= 0 for t in times):
            raise ValueError(f"all exercise_times must be positive, got {times}")
        if any(t > maturity + _TIME_TOLERANCE for t in times):
            raise ValueError(f"exercise_times must not exceed maturity {maturity}, got {times}")
        self.exercise_times = sorted(set(times) | {maturity})
 
    def is_exercisable(self, state: PricingState) -> bool:
        current_time = state.time
        return any(_same_time(current_time, t) for t in self.exercise_times)

    def check_grid(self, n_steps: int) -> None:
        for t in self.exercise_times:
            grid_point = t * n_steps / self.maturity
            if abs(grid_point - round(grid_point)) > _GRID_TOLERANCE:
                raise ValueError(
                    f"exercise time {t} is not on the {n_steps}-step grid "
                    f"(dt={self.maturity / n_steps}); it would never be exercised. "
                    f"Choose n_steps so that every exercise time is a multiple of dt."
                )