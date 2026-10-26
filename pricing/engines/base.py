from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from pricing.contract import DerivativeContract
from pricing.markets import MarketData
from pricing.stochastic_process import  StochasticProcess


@dataclass(frozen=True)
class PricingResult:
    """
    Container for the output of an engine.
    Includes a price with its uncertainty and diagnostics.
    """
    price: float
    std_error: float | None = None      # None for deterministic engines
    runtime: float = 0.0                # seconds
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ci95(self) -> tuple[float, float] | None:
        if self.std_error is None:
            return None
        return (self.price - 1.96 * self.std_error, self.price + 1.96 * self.std_error)

    def __float__(self) -> float:
        return float(self.price)
    

class PricingEngine(ABC):
    """Common interface for all pricing engines."""

    def check_supported(self, derivative: DerivativeContract,
                         process: StochasticProcess) -> None:
        """Raise NotImplementedError for (derivative, process) pairs the engine
        cannot price correctly. Default: everything is supported."""

    @abstractmethod
    def price(self, derivative: DerivativeContract, process: StochasticProcess,
              initial_spot: float, market_data: MarketData) -> PricingResult:
        """Return the present value of `derivative` under `process`/`market_data`."""