"""Theta-scheme finite-difference engine for the Black-Scholes PDE."""
from __future__ import annotations
 
import numbers
import time
from functools import partial
from typing import Any
 
import numpy as np
 
from pricing.contract import DerivativeContract
from pricing.engines.base import PricingEngine, PricingResult
from pricing.exercise_rights import AmericanExercise, BermudanExercise
from pricing.markets import MarketData
from pricing.payoff import VanillaPayoff
from pricing.state import PathDependentState, PricingState, UnderlyingState
from pricing.stochastic_process import GeometricBrownianMotion, StochasticProcess

class FiniteDifferenceEngine(PricingEngine):
    """Theta-scheme finite-difference engine for vanilla options under GBM.

    The spatial grid may be uniform in spot or log-spot. ``theta`` selects
    explicit Euler (0), Crank-Nicolson (0.5), or implicit Euler (1).

    This engine supports vanilla European, American, and grid-aligned
    Bermudan exercise. It does not support path-dependent payoffs.

    Args:
        n_space: number of space intervals (integer >= 3).
        n_time: number of time steps (positive integer).
        theta: time-stepping weight in [0, 1].
        grid_type: 'spot' (uniform in S) or 'log' (uniform in log S).
        s_max: upper spot bound of a 'spot' grid. Default: 3 * max(S0, strike).
        log_width_sigma: half-width of a 'log' grid, in standard deviations of
            the log-spot at maturity.
 
    Raises:
        ValueError: on invalid arguments.
    """

    def __init__(
        self,
        n_space: int = 400,
        n_time: int = 400,
        theta: float = 0.5,
        grid_type: str = "log",
        s_max: float | None = None,
        log_width_sigma: float = 5.0,
    ):
        if not isinstance(n_space, numbers.Integral) or n_space < 3:
            raise ValueError("n_space must be an integer >= 3")
        if not isinstance(n_time, numbers.Integral) or n_time <= 0:
            raise ValueError("n_time must be a positive integer")
        if not np.isfinite(theta) or not 0.0 <= theta <= 1.0:
            raise ValueError("theta must be finite and between 0 and 1")
        if grid_type not in {"spot", "log"}:
            raise ValueError("grid_type must be 'spot' or 'log'")
        if s_max is not None and (not np.isfinite(s_max) or s_max <= 0):
            raise ValueError("s_max must be finite and positive when provided")
        if not np.isfinite(log_width_sigma) or log_width_sigma <= 0:
            raise ValueError("log_width_sigma must be finite and positive")

        self.n_space = int(n_space)
        self.n_time = int(n_time)
        self.theta = float(theta)
        self.grid_type = grid_type
        self.s_max = None if s_max is None else float(s_max)
        self.log_width_sigma = float(log_width_sigma)

    def check_supported(
        self,
        derivative: DerivativeContract,
        process: StochasticProcess,
    ) -> None:
        if not isinstance(process, GeometricBrownianMotion):
            raise NotImplementedError("FiniteDifferenceEngine assumes a GBM process.")
        if not isinstance(derivative.payoff_fn, VanillaPayoff):
            raise NotImplementedError(
                "FiniteDifferenceEngine currently supports vanilla call/put payoffs only."
            )

    def build_spot_grid(
        self,
        initial_spot: float,
        strike: float,
        maturity: float,
        volatility: float,
        risk_free_rate: float,
        dividend_yield: float,
    ) -> np.ndarray:
        """Build and return the configured spatial grid in spot coordinates."""
        if self.grid_type == "spot":
            upper_bound = self.s_max or 3.0 * max(initial_spot, strike)
            if initial_spot >= upper_bound:
                raise ValueError("initial_spot must be below s_max for a spot grid")
            return np.linspace(0.0, upper_bound, self.n_space + 1)

        drift = risk_free_rate - dividend_yield - 0.5 * volatility**2
        half_width = max(
            self.log_width_sigma * volatility * np.sqrt(maturity)
            + abs(drift) * maturity,
            abs(np.log(strike / initial_spot)) + 1.0 / self.n_space,
        )
        log_grid = np.linspace(
            np.log(initial_spot) - half_width,
            np.log(initial_spot) + half_width,
            self.n_space + 1,
        )
        return np.exp(log_grid)

    @staticmethod
    def _solve_tridiagonal(
        lower: np.ndarray,
        diagonal: np.ndarray,
        upper: np.ndarray,
        rhs: np.ndarray,
    ) -> np.ndarray:
        """Solve a tridiagonal system using the Thomas algorithm."""
        modified_diagonal = diagonal.astype(float, copy=True)
        modified_rhs = rhs.astype(float, copy=True)
        upper = upper.astype(float, copy=False)
        lower = lower.astype(float, copy=False)

        for row in range(1, modified_diagonal.size):
            pivot = modified_diagonal[row - 1]
            if not np.isfinite(pivot) or abs(pivot) <= np.finfo(float).eps:
                raise np.linalg.LinAlgError("Singular tridiagonal PDE system.")
            multiplier = lower[row - 1] / pivot
            modified_diagonal[row] -= multiplier * upper[row - 1]
            modified_rhs[row] -= multiplier * modified_rhs[row - 1]

        if (
            not np.isfinite(modified_diagonal[-1])
            or abs(modified_diagonal[-1]) <= np.finfo(float).eps
        ):
            raise np.linalg.LinAlgError("Singular tridiagonal PDE system.")

        solution = np.empty_like(modified_rhs)
        solution[-1] = modified_rhs[-1] / modified_diagonal[-1]
        for row in range(modified_diagonal.size - 2, -1, -1):
            solution[row] = (
                modified_rhs[row] - upper[row] * solution[row + 1]
            ) / modified_diagonal[row]
        return solution

    def price(
        self,
        derivative: DerivativeContract,
        process: StochasticProcess,
        initial_spot: float,
        market_data: MarketData,
    ) -> PricingResult:
        started = time.perf_counter()
        self.check_supported(derivative, process)
        derivative.validate_grid(self.n_time)

        payoff = derivative.payoff_fn
        strike = payoff.strike
        if not np.isfinite(initial_spot) or initial_spot <= 0:
            raise ValueError("initial_spot must be finite and positive")
        if not np.isfinite(strike) or strike <= 0:
            raise ValueError("strike must be finite and positive")

        maturity = derivative.maturity
        dt = maturity / self.n_time
        sigma = process.volatility
        rate = market_data.risk_free_rate
        dividend = market_data.dividend_yield
        spots = self.build_spot_grid(
            initial_spot, strike, maturity, sigma, rate, dividend
        )
        if initial_spot >= spots[-1]:
            raise ValueError("initial_spot must be inside the spatial grid")
        interior_spots = spots[1:-1]

        if self.grid_type == "log":
            log_spots = np.log(spots)
            spacing = float(log_spots[1] - log_spots[0])
            drift = rate - dividend - 0.5 * sigma**2
            lower_operator = (
                0.5 * sigma**2 / spacing**2 - 0.5 * drift / spacing
            )
            diagonal_operator = -sigma**2 / spacing**2 - rate
            upper_operator = (
                0.5 * sigma**2 / spacing**2 + 0.5 * drift / spacing
            )
            lower = np.full(self.n_space - 1, lower_operator)
            diagonal = np.full(self.n_space - 1, diagonal_operator)
            upper = np.full(self.n_space - 1, upper_operator)
        else:
            spacing = float(spots[1] - spots[0])
            interior_index = np.arange(1, self.n_space, dtype=float)
            lower = (
                0.5 * sigma**2 * interior_index**2
                - 0.5 * (rate - dividend) * interior_index
            )
            diagonal = -sigma**2 * interior_index**2 - rate
            upper = (
                0.5 * sigma**2 * interior_index**2
                + 0.5 * (rate - dividend) * interior_index
            )

        theta = self.theta
        lhs_lower = -theta * dt * lower[1:]
        lhs_diagonal = 1.0 - theta * dt * diagonal
        lhs_upper = -theta * dt * upper[:-1]
        rhs_diagonal = 1.0 + (1.0 - theta) * dt * diagonal
        rhs_lower = (1.0 - theta) * dt * lower[1:]
        rhs_upper = (1.0 - theta) * dt * upper[:-1]

        early_exercise = isinstance(
            derivative.exercise_rights, (AmericanExercise, BermudanExercise)
        )

        def boundary_values(time_value: float) -> tuple[float, float]:
            tau = maturity - time_value
            if early_exercise:
                if payoff.is_call:
                    return 0.0, max(spots[-1] - strike, 0.0)
                return strike, 0.0
            if payoff.is_call:
                high = max(
                    spots[-1] * np.exp(-dividend * tau)
                    - strike * np.exp(-rate * tau),
                    0.0,
                )
                return 0.0, float(high)
            low = strike * np.exp(-rate * tau)
            return float(low), 0.0

        path_state = PathDependentState()

        def state_at(time_value: float, spot_values: np.ndarray) -> PricingState:
            return PricingState(
                UnderlyingState(time=time_value, spot=spot_values),
                path_state,
            )

        values = np.asarray(
            derivative.payoff(state_at(maturity, spots)), dtype=float
        ).copy()
        minimum_explicit_diagonal_weight = float(
            np.min(1.0 + dt * diagonal)
        )
        minimum_explicit_off_diagonal_weight = float(
            min(np.min(dt * lower), np.min(dt * upper))
        )
        explicit_weights_nonnegative = (
            minimum_explicit_diagonal_weight >= 0.0
            and minimum_explicit_off_diagonal_weight >= 0.0
        )

        for step in range(self.n_time):
            old_time = maturity - step * dt
            new_time = maturity - (step + 1) * dt
            old_low, old_high = boundary_values(old_time)
            new_low, new_high = boundary_values(new_time)

            with np.errstate(over="ignore", invalid="ignore"):
                rhs = rhs_diagonal * values[1:-1]
                rhs[1:] += rhs_lower * values[1:-2]
                rhs[:-1] += rhs_upper * values[2:-1]
                rhs[0] += dt * lower[0] * (
                    (1.0 - theta) * old_low + theta * new_low
                )
                rhs[-1] += dt * upper[-1] * (
                    (1.0 - theta) * old_high + theta * new_high
                )
            if not np.all(np.isfinite(rhs)):
                raise FloatingPointError(
                    "Finite-difference right-hand side became non-finite; "
                    "reduce the time step or refine the spatial grid."
                )

            if theta == 0.0:
                next_values = rhs
            else:
                next_values = self._solve_tridiagonal(
                    lhs_lower, lhs_diagonal, lhs_upper, rhs
                )

            state = state_at(new_time, interior_spots)
            if derivative.is_exercisable(state):
                next_values = np.maximum(
                    next_values, derivative.payoff(state)
                )
            if not np.all(np.isfinite(next_values)):
                raise FloatingPointError(
                    "Finite-difference values became non-finite; reduce the "
                    "time step or refine the spatial grid."
                )

            values[0] = new_low
            values[-1] = new_high
            values[1:-1] = next_values

        price = float(np.interp(initial_spot, spots, values))
        if derivative.is_exercisable(state_at(0.0, np.asarray(initial_spot))):
            price = max(
                price,
                float(derivative.payoff(state_at(0.0, np.asarray(initial_spot)))),
            )

        metadata: dict[str, Any] = {
            "engine": "finite_difference_theta",
            "theta": theta,
            "grid_type": self.grid_type,
            "n_space": self.n_space,
            "n_time": self.n_time,
            "explicit_weights_nonnegative": explicit_weights_nonnegative,
            "minimum_explicit_diagonal_weight": minimum_explicit_diagonal_weight,
            "minimum_explicit_off_diagonal_weight": minimum_explicit_off_diagonal_weight,
        }
        if self.grid_type == "log":
            metadata["log_spacing"] = spacing
        else:
            metadata["spot_spacing"] = spacing

        return PricingResult(
            price=price,
            std_error=None,
            runtime=time.perf_counter() - started,
            metadata=metadata,
        )
