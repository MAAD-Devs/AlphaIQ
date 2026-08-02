"""
Specialized financial analytics engines.
"""

from .annuities import AnnuityMonteCarloPricer, IndexedAnnuityPayoff
from .factor_models import FamaFrench5Factor, MarcenkoPasturDenoiser
from .fixed_income import BondAnalytics, SVFYieldCurve
from .risk_metrics import (
    InformationRatio,
    SharpeRatio,
    SortinoRatio,
    TreynorRatio,
    compute_all_risk_metrics,
)
from .sector_analysis import SectorRotatorML
from .tail_risk import (
    CalmarRatio,
    ConditionalVaR,
    MaximumDrawdown,
    UlcerIndex,
    ValueAtRisk,
)

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
