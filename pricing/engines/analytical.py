"""Closed-form Black-Scholes engine for European vanilla and digital options."""
from __future__ import annotations
 
import math
import time
 
from pricing.contract import DerivativeContract
from pricing.engines.base import PricingEngine, PricingResult
from pricing.exercise_rights import EuropeanExercise
from pricing.markets import MarketData
from pricing.payoff import DigitalPayoff, VanillaPayoff
from pricing.stochastic_process import GeometricBrownianMotion, StochasticProcess

def _d1_d2(initial_spot: float, strike: float, risk_free_rate: float, volatility: float, maturity: float, dividend_yield: float) -> tuple[float, float]: 
        d1 = (math.log(initial_spot/strike) + (risk_free_rate-dividend_yield+0.5*volatility**2)*maturity)/(volatility*math.sqrt(maturity))
        d2 = d1 - volatility*math.sqrt(maturity)
        return d1, d2

def _cfd(x: float):
    return 0.5*(1 + math.erf(x/math.sqrt(2)))

def _price_vanilla(initial_spot: float, strike: float, risk_free_rate: float, 
                    dividend_yield: float, maturity: float, d1: float, d2: float, 
                    is_call: bool) -> float:
    pv_strike = strike*math.exp(-risk_free_rate*maturity)
    if is_call:
        return initial_spot*math.exp(-dividend_yield*maturity) * _cfd(d1) - pv_strike*_cfd(d2)
    return -initial_spot*math.exp(-dividend_yield*maturity) * _cfd(-d1) + pv_strike*_cfd(-d2)

def _price_digital(cash_payout: float, risk_free_rate: float, maturity: float,
                    d2: float, is_call: bool) -> float:
    if is_call:
        return cash_payout*math.exp(-risk_free_rate*maturity) * _cfd(d2)
    return cash_payout*math.exp(-risk_free_rate*maturity) * _cfd(-d2)

class AnalyticalEngine(PricingEngine):

    def check_supported(self, derivative: DerivativeContract,
                            process: StochasticProcess) -> None:
        if not isinstance(process, GeometricBrownianMotion):
            raise NotImplementedError("Black Scholes assumes a GBM process.")
        if not isinstance(derivative.exercise_rights, EuropeanExercise):
            raise NotImplementedError(
                "BlackScholesEngine only prices European exercise; use "
                "BinomialEngine or FiniteDifferenceEngine for early exercise."
            )
        if not isinstance(derivative.payoff_fn, (VanillaPayoff, DigitalPayoff)):
            raise NotImplementedError(
                "BlackScholesEngine supports VanillaPayoff and DigitalPayoff only."
            )
        
    def price(self, derivative: DerivativeContract, process: StochasticProcess,
                  initial_spot: float, market_data: MarketData) -> PricingResult:

        started = time.perf_counter()
        self.check_supported(derivative, process)
        if not math.isfinite(initial_spot) or initial_spot <= 0:
            raise ValueError("initial_spot must be finite and positive")
 
        strike = derivative.payoff_fn.strike
        maturity = derivative.maturity
        risk_free_rate = market_data.risk_free_rate
        dividend_yield = market_data.dividend_yield
        volatility = process.volatility
        is_call = derivative.payoff_fn.is_call
        d1, d2 = _d1_d2(initial_spot, strike, risk_free_rate, volatility, maturity, dividend_yield)

        if isinstance(derivative.payoff_fn, VanillaPayoff):
            price = _price_vanilla(initial_spot, strike, risk_free_rate, dividend_yield, maturity, d1, d2, is_call)
        elif isinstance(derivative.payoff_fn, DigitalPayoff):
            price = _price_digital(derivative.payoff_fn.cash_payout, risk_free_rate, maturity, d2, is_call)

        return PricingResult(
            price=price,
            std_error=None,
            runtime=time.perf_counter() - started,
            metadata={"engine": "analytical_black_scholes"},
        )