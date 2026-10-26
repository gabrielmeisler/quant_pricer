from pricing.payoff import Payoff
from pricing.exercise_rights import ExerciseRights
from pricing.state import PricingState
from pricing.path_tracker import PathFeatureTracker, collect_trackers_by_name

class DerivativeContract:
    """
    A derivative defined by its payoff and exercise rights. Path trackers required by 
    the payoff are automatically validated against registered trackers and instantiated 
    once at construction time.

    Args:
        payoff (Payoff): The payoff function that declares required trackers and configs.
        exercise_rights (ExerciseRights): The exercise rights that define maturity and exercise rules.
    """

    def __init__(self, payoff: Payoff, exercise_rights: ExerciseRights):        
        self.payoff_fn = payoff
        self.exercise_rights = exercise_rights
        self.maturity = exercise_rights.maturity

        required = payoff.required_path_trackers
        available = PathFeatureTracker.registered_names()
        if not required.issubset(available):
            raise ValueError(
                f"Payoff requires trackers {set(required - available)}, "
                f"but only {set(available)} are available"
            )
        self._path_trackers = collect_trackers_by_name(required, payoff.tracker_configs)
    
    def payoff(self, state: PricingState):
            return self.payoff_fn(state)
    
    def is_exercisable(self, state: PricingState) -> bool:
        return self.exercise_rights.is_exercisable(state)

    def validate_grid(self, n_steps: int) -> None:
        """Raise if exercise dates cannot be represented on an `n_steps` grid."""
        self.exercise_rights.check_grid(n_steps)

    @property
    def required_path_trackers(self) -> frozenset:
        return self.payoff_fn.required_path_trackers

    @property
    def path_trackers(self) -> list[PathFeatureTracker]:
        return list(self._path_trackers)

    @property
    def is_path_dependent(self) -> bool:
        return bool(self._path_trackers)