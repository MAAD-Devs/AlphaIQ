"""
Specialized financial analytics engines.
"""

from .risk_metrics import SharpeRatio, SortinoRatio, TreynorRatio, InformationRatio, compute_all_risk_metrics
from .tail_risk import ValueAtRisk, ConditionalVaR, MaximumDrawdown, CalmarRatio, UlcerIndex
from .factor_models import FamaFrench5Factor, MarcenkoPasturDenoiser
from .fixed_income import BondAnalytics, SVFYieldCurve
from .annuities import IndexedAnnuityPayoff, AnnuityMonteCarloPricer
from .sector_analysis import SectorRotatorML

__all__ = [
    "SharpeRatio",
    "SortinoRatio",
    "TreynorRatio",
    "InformationRatio",
    "compute_all_risk_metrics",
    "ValueAtRisk",
    "ConditionalVaR",
    "MaximumDrawdown",
    "CalmarRatio",
    "UlcerIndex",
    "FamaFrench5Factor",
    "MarcenkoPasturDenoiser",
    "BondAnalytics",
    "SVFYieldCurve",
    "IndexedAnnuityPayoff",
    "AnnuityMonteCarloPricer",
    "SectorRotatorML",
]
