from dataclasses import dataclass

# TODO: UNDERSTANF WHY DIVIDEND YIELD MAY BE NEGATIVE

@dataclass
class MarketData:
    """
    Flat, constant-in-time market data for a single underlying.
 
    Args:
        risk_free_rate: Continuously-compounded risk-free rate (e.g. 0.05
            for 5%). May be negative.
        dividend_yield: Continuously-compounded dividend/yield rate on the
            underlying. May be negative (e.g. a foreign rate for FX, or a
            convenience yield for commodities).
    """
    risk_free_rate: float
    dividend_yield: float
