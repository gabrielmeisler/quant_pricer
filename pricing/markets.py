from dataclasses import dataclass

@dataclass
class MarketData:
    """
    Flat, constant-in-time market data for a single underlying.
 
    Args:
        risk_free_rate: Continuously-compounded risk-free rate.
        dividend_yield: Continuously-compounded dividend/yield rate on the underlying. 
    """
    risk_free_rate: float
    dividend_yield: float
