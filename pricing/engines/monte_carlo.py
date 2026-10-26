"""Monte Carlo pricing engine with Longstaff-Schwartz for early exercise."""

from __future__ import annotations
 
import numbers
import time
from collections.abc import Callable
 
import numpy as np
 
from pricing.contract import DerivativeContract
from pricing.engines.base import PricingEngine, PricingResult
from pricing.markets import MarketData
from pricing.path_tracker import build_pricing_path
from pricing.state import PricingState
from pricing.stochastic_process import StochasticProcess

LSMBasisFunction = Callable[[PricingState, np.ndarray, float], np.ndarray]

def polynomial_basis(degree: int) -> Callable[[PricingState, np.ndarray, float], np.ndarray]:
    """
    Default LSM regression basis: polynomial of degree `degree` in x = S/S0,
    plus (a, a^2, a*x) for every continuous path feature `a` (normalised by S0),
    e.g. the running average of an Asian option. Boolean features (knock-out
    flags) are skipped: knocked paths have zero payoff and never enter the
    in-the-money regression set.
    """
    def basis(state: PricingState, mask: np.ndarray, scale: float) -> np.ndarray:
        x = np.asarray(state.spot)[mask] / scale
        columns = [x ** k for k in range(degree + 1)]
        for value in state.path_state.params.values():
            arr = np.asarray(value)
            if arr.dtype.kind != "f":
                continue
            a = arr[mask] / scale
            columns += [a, a ** 2, a * x]
        return np.column_stack(columns)
    return basis


class MonteCarloEngine(PricingEngine):
    """
    Monte Carlo pricer. Early exercise is handled by Longstaff-Schwartz (LSM).

    Args:
        n_paths: number of paths (positive integer).
        n_steps: time steps per path (positive integer).
        seed: optional seed for reproducibility.
        poly_degree: degree of the polynomial-in-spot part of the LSM basis (>= 1).
        basis: optional custom regression basis, `f(state, mask, scale) -> (n_itm, n_features)`.

    Raises:
        ValueError: on invalid arguments.
    """

    def __init__(self, n_paths: int, n_steps: int, seed: int | None = None,
                 poly_degree: int = 2, basis: Callable[[PricingState, np.ndarray, float], np.ndarray] | None = None):
        if not isinstance(n_paths, numbers.Integral) or n_paths <= 1:
            raise ValueError("n_paths must be an integer >= 2")
        if not isinstance(n_steps, numbers.Integral) or n_steps <= 0:
            raise ValueError("n_steps must be a positive integer")
        if not isinstance(poly_degree, numbers.Integral) or poly_degree < 1:
            raise ValueError(f"poly_degree must be >= 1, got {poly_degree}")
        self.n_paths = int(n_paths)
        self.n_steps = int(n_steps)
        self.seed = seed
        self.poly_degree = int(poly_degree)
        self.basis = basis or polynomial_basis(self.poly_degree)

    def price(self, derivative: DerivativeContract, process: StochasticProcess, initial_spot: float, 
              market_data: MarketData) -> PricingResult:
        started = time.perf_counter()
        self.check_supported(derivative, process)
        derivative.validate_grid(self.n_steps)

        dt = derivative.maturity / self.n_steps
        discount = np.exp(-market_data.risk_free_rate * dt)

        # Generation of simulation paths
        underlying = process.simulate_paths(
            initial_spot=initial_spot, maturity=derivative.maturity,
            n_steps=self.n_steps, n_paths=self.n_paths,
            market_data=market_data, seed=self.seed,
        )
        paths = build_pricing_path(underlying, derivative.path_trackers)

        # Pathwise cashflow-to-go initialization at maturity
        cashflow_to_go = np.array(derivative.payoff(paths[-1]), dtype=float)
        for i in range(self.n_steps - 1, 0, -1):

            # Holding value backward induction.
            cashflow_to_go = cashflow_to_go * discount
            state = paths[i]
            if not derivative.is_exercisable(state):
                continue  # not an exercise date: holding is the only option

            # Collecting in-the-money paths
            exercise = np.asarray(derivative.payoff(state), dtype=float)
            itm = exercise > 0.0
            if not np.any(itm):
                continue

            # Estimate the continuation value E[cashflow-to-go | state] by regressing
            # the realised (discounted) future cashflows on the basis functions.
            features = self.basis(state, itm, float(initial_spot))
            if features.shape[0] <= features.shape[1]:
                continue  # continue if regression is underdetermined
            coeffs, *_ = np.linalg.lstsq(features, cashflow_to_go[itm], rcond=None)
            continuation = features @ coeffs

            # Exercise where immediate payoff beats the estimated continuation value.
            # For those paths, the exercise payoff replaces the future cashflow
            exercise_now = exercise[itm] > continuation
            idx = np.flatnonzero(itm)[exercise_now] 
            cashflow_to_go[idx] = exercise[idx]

        # The initial time is excluded from the loop because every path starts at the same spot,
        # so there is nothing to regress. The exercise decision is simply
        # "exercise now" versus the Monte Carlo continuation estimate.
        samples = cashflow_to_go * discount
        price = float(samples.mean())
        if derivative.is_exercisable(paths[0]):
            price = max(price, float(np.mean(derivative.payoff(paths[0]))))

        std_error = float(samples.std(ddof=1) / np.sqrt(self.n_paths))
        return PricingResult(
            price=price, std_error=std_error,
            runtime=time.perf_counter() - started,
            metadata={"engine": "monte_carlo", "n_paths": self.n_paths,
                      "n_steps": self.n_steps, "seed": self.seed},
        )
