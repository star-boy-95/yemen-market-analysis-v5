"""
Pydantic schemas for data validation in Yemen Market Analysis.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Literal
from pydantic import BaseModel, Field, validator, root_validator
from datetime import datetime


class PriceSeries(BaseModel):
    """Schema for price time series data."""
    commodity: str
    regime: str
    dates: List[datetime]
    prices: List[float]
    
    @validator('prices')
    def check_prices(cls, v):
        """Validate prices."""
        if not v:
            raise ValueError("Empty price list")
        if any(p <= 0 for p in v):
            raise ValueError("Prices must be positive")
        return v
    
    @validator('dates')
    def check_dates(cls, v):
        """Validate dates."""
        if not v:
            raise ValueError("Empty date list")
        if any(not isinstance(d, datetime) for d in v):
            raise ValueError("All dates must be datetime objects")
        return v
    
    def to_series(self) -> pd.Series:
        """Convert to pandas Series."""
        return pd.Series(self.prices, index=self.dates)
    
    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True


class ModelConfig(BaseModel):
    """Schema for threshold model configuration."""
    grid_points: int = Field(50, ge=1)
    nboot: int = Field(500, ge=100)
    block_size: int = Field(5, ge=1)
    min_regime_size: float = Field(0.1, ge=0.05, le=0.5)
    max_lags: int = Field(8, ge=1)
    use_adaptive_thresholds: bool = True
    significance_level: float = Field(0.05, gt=0, lt=0.5)
    conflict_adjustment_factor: float = Field(0.5, ge=0, le=1)
    parallel_processing: bool = Field(False)
    use_gpu: bool = Field(False)
    model_type: str = Field("hansen_seo")
    ic_type: str = Field("aic")
    early_stopping: bool = Field(True)
    use_hac: bool = Field(True)
    hac_maxlags: Optional[int] = None
    
    class Config:
        """Pydantic config."""
        validate_assignment = True


class ThresholdRange(BaseModel):
    """Schema for threshold ranges."""
    commodity: str
    lower: float
    upper: float
    
    @validator('upper')
    def check_upper_greater_than_lower(cls, v, values):
        """Validate upper > lower."""
        if 'lower' in values and v <= values['lower']:
            raise ValueError("Upper threshold must be greater than lower threshold")
        return v


class ThresholdResult(BaseModel):
    """Schema for threshold model results."""
    commodity: str
    threshold: float
    test_statistic: float
    p_value: Optional[float] = None
    threshold_significant: bool = False
    alpha_up: Optional[float] = None
    alpha_down: Optional[float] = None
    regime_balance: Dict[str, float] = Field(default_factory=dict)
    half_life_up: Optional[float] = None
    half_life_down: Optional[float] = None
    asymmetry_significant: bool = False
    asymmetry_pvalue: Optional[float] = None
    model_type: str
    integration_index: Optional[float] = None
    integration_level: Optional[str] = None
    
    class Config:
        """Pydantic config."""
        validate_assignment = True


class MarketData(BaseModel):
    """Schema for market data."""
    admin1: str
    commodity: str
    date: datetime
    price: float = Field(..., gt=0)
    usdprice: float = Field(..., gt=0)
    exchange_rate_regime: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    conflict_intensity: Optional[float] = None
    population: Optional[int] = None
    
    @validator('exchange_rate_regime')
    def check_regime(cls, v):
        """Validate exchange rate regime."""
        if v not in ('north', 'south'):
            raise ValueError("Exchange rate regime must be 'north' or 'south'")
        return v
    
    @validator('price', 'usdprice')
    def check_positive(cls, v):
        """Validate positive prices."""
        if v <= 0:
            raise ValueError("Price must be positive")
        return v
    
    @validator('admin1')
    def check_admin1(cls, v):
        """Validate admin1 region name."""
        if not v:
            raise ValueError("Admin1 region name cannot be empty")
        return v.lower()  # Standardize to lowercase


class ValidationResult(BaseModel):
    """Schema for validation results."""
    status: Literal['passed', 'warning', 'failed']
    commodity: str
    n_observations: int
    valid_observations: int
    missing_values: int
    outliers: int
    economic_violations: Dict[str, Any] = Field(default_factory=dict)
    messages: List[str] = Field(default_factory=list)


class ConflictData(BaseModel):
    """Schema for conflict intensity data."""
    admin1: str
    date: datetime
    conflict_intensity: float = Field(..., ge=0)
    events: Optional[int] = None
    fatalities: Optional[int] = None
    
    @validator('conflict_intensity')
    def check_conflict_intensity(cls, v):
        """Validate conflict intensity."""
        if v < 0:
            raise ValueError("Conflict intensity cannot be negative")
        return v


class WelfareAnalysisResult(BaseModel):
    """Schema for welfare analysis results."""
    commodity: str
    total_deadweight_loss: float
    dwl_percent_of_market: float
    arbitrage_frequency: float
    arbitrage_days: int
    average_price_differential: float
    max_price_differential: float
    north_higher_pct: float
    south_higher_pct: float
    surplus_direction: str
    net_welfare_impact: str
    welfare_impact_severity: str
    
    @root_validator
    def check_percentages(cls, values):
        """Validate percentages sum to 100."""
        north = values.get('north_higher_pct', 0)
        south = values.get('south_higher_pct', 0)
        if abs(north + south - 100) > 1:
            raise ValueError("North and South percentages should approximately sum to 100%")
        return values


class PolicyImpact(BaseModel):
    """Schema for policy impact analysis."""
    commodity: str
    threshold_reduction: float
    welfare_gain: float
    welfare_gain_pct: float
    arbitrage_increase: float
    trade_volume_increase: float
    implementation_cost: Optional[float] = None
    benefit_cost_ratio: Optional[float] = None
    
    @validator('threshold_reduction')
    def check_threshold_reduction(cls, v):
        """Validate threshold reduction."""
        if not 0 <= v <= 1:
            raise ValueError("Threshold reduction must be between 0 and 1")
        return v


class RegionalAnalysis(BaseModel):
    """Schema for regional analysis results."""
    region: str
    commodity: str
    price_north: float
    price_south: float
    price_differential: float
    price_differential_pct: float
    arbitrage_opportunity: bool
    welfare_impact: float
    
    @validator('price_north', 'price_south')
    def check_prices(cls, v):
        """Validate prices."""
        if v <= 0:
            raise ValueError("Prices must be positive")
        return v