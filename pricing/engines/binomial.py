"""Cox-Ross-Rubinstein binomial tree engine."""
from __future__ import annotations
 
import numbers
import time
import numpy as np
 
from pricing.contract import DerivativeContract
from pricing.engines.base import PricingEngine, PricingResult
from pricing.markets import MarketData
from pricing.state import PathDependentState, PricingState, UnderlyingState
from pricing.stochastic_process import GeometricBrownianMotion, StochasticProcess
             
class BinomialEngine(PricingEngine):
    """
    Cox-Ross-Rubinstein recombining tree for GBM.

    Supports European / Bermudan / American exercise on payoffs that depend on
    the spot only. Path-dependent payoffs are rejected. Future iterations will
    be extended to support Asian options through the Hull-White Model.

    Args:
        tree_depth: number of tree steps (positive integer).
    """

    def __init__(self, tree_depth: int):
        if not isinstance(tree_depth, numbers.Integral) or tree_depth <= 0:
            raise ValueError("tree_depth must be a positive integer")
        self.tree_depth = int(tree_depth)

    def check_supported(self, derivative: DerivativeContract,
                        process: StochasticProcess) -> None:
        if not isinstance(process, GeometricBrownianMotion):
            raise NotImplementedError("BinomialEngine assumes a GBM process.")
        if derivative.is_path_dependent:
            raise NotImplementedError(
                f"BinomialEngine cannot price path-dependent payoffs "
                f"(requires trackers {sorted(derivative.required_path_trackers)}): "
                f"use MonteCarloEngine, or a non-recombining / augmented-state tree."
            )

    def price(self, derivative: DerivativeContract, process: StochasticProcess,
              initial_spot: float, market_data: MarketData) -> PricingResult:
        started = time.perf_counter()
        self.check_supported(derivative, process)
        n = self.tree_depth
        derivative.validate_grid(n)

        r, q = market_data.risk_free_rate, market_data.dividend_yield
        dt = derivative.maturity / n
        u = np.exp(process.volatility * np.sqrt(dt))
        d = 1.0 / u
        p = (np.exp((r - q) * dt) - d) / (u - d)
        if not 0.0 < p < 1.0:
            raise ValueError(
                f"risk-neutral probability p={p:.4f} is outside (0, 1): "
                f"increase tree_depth or check volatility vs (r - q)."
            )
        discount = np.exp(-r * dt)
        no_path_state = PathDependentState()

        def state_at(level: int) -> PricingState:
            # Since u=1/d, node k at level i has spot S0 * u^(2k - i)
            spots = initial_spot * u ** (2.0 * np.arange(level + 1) - level)
            return PricingState(UnderlyingState(time=level * dt, spot=spots), no_path_state)

        # Backwards induction
        values = np.asarray(derivative.payoff(state_at(n)), dtype=float)  # maturity
        for level in range(n - 1, -1, -1):
            values = discount * (p * values[1:] + (1.0 - p) * values[:-1])
            state = state_at(level)
            if derivative.is_exercisable(state):      # once per level, not per node
                values = np.maximum(values, derivative.payoff(state))

        return PricingResult(
            price=float(values[0]), std_error=None,
            runtime=time.perf_counter() - started,
            metadata={"engine": "binomial_crr", "tree_depth": n},
        )
