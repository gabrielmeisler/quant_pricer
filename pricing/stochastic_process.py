from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Any
from pricing.state import UnderlyingState
from pricing.markets import MarketData
import numpy as np


class StochasticProcess(ABC):
    """
    Abstract model of the underlying. It knows nothing about contracts or
    trackers: it only produces underlying paths. Path-dependent features are
    applied afterwards by `path_tracker.build_pricing_path`.
    """

    @property
    @abstractmethod
    def n_factors(self) -> int:
        """Number of independent Brownian factors driving the model."""

    @abstractmethod
    def simulation_step(self, current_model_state: UnderlyingState, dt: float,
                        z, market_data: MarketData) -> UnderlyingState:
        """Advance one step with standard-normal shocks `z`."""

    @abstractmethod
    def simulate_paths(self, initial_spot: float, maturity: float, n_steps: int,
                       n_paths: int, market_data: MarketData,
                       seed: int | None = None) -> list[UnderlyingState]:
        """Return n_steps + 1 states (t=0 included), each holding all paths."""


class GeometricBrownianMotion(StochasticProcess):
    """dS = (r - q) S dt + sigma S dW, simulated with the exact log-Euler step."""

    n_factors = 1

    def __init__(self, volatility: float):
        if volatility <= 0:
            raise ValueError(f"volatility must be positive, got {volatility}")
        self.volatility = volatility

    def simulation_step(self, current_model_state: UnderlyingState, dt: float,
                            z, market_data: MarketData) -> UnderlyingState:
        r, q, sigma = market_data.risk_free_rate, market_data.dividend_yield, self.volatility
        drift = (r - q - 0.5 * sigma ** 2) * dt
        diffusion = sigma * np.sqrt(dt) * z
        return UnderlyingState(
            time=current_model_state.time + dt,
            spot=current_model_state.spot * np.exp(drift + diffusion),
        )

    def simulate_paths(self, initial_spot: float, maturity: float, n_steps: int,
                        n_paths: int, market_data: MarketData,
                        seed: int | None = None) -> list[UnderlyingState]:
        times = np.linspace(0.0, maturity, n_steps + 1)  
        dt = maturity / n_steps
        rng = np.random.default_rng(seed)

        state = UnderlyingState(time=0.0, spot=np.full(n_paths, float(initial_spot)))
        history = [state]
        for k in range(n_steps):
            z = rng.normal(size=n_paths)
            nxt = self.simulation_step(state, dt, z, market_data)
            state = replace(nxt, time=float(times[k + 1]))
            history.append(state)
        return history