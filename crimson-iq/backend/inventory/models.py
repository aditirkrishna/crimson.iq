# Pydantic + SQLAlchemy models for DB and API schemas

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, validator
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import uuid

Base = declarative_base()

# Enums
class BloodGroup(str, Enum):
    A_POSITIVE = "A+"
    A_NEGATIVE = "A-"
    B_POSITIVE = "B+"
    B_NEGATIVE = "B-"
    AB_POSITIVE = "AB+"
    AB_NEGATIVE = "AB-"
    O_POSITIVE = "O+"
    O_NEGATIVE = "O-"

class BloodUnitStatus(str, Enum):
    AVAILABLE = "available"
    RESERVED = "reserved"
    TRANSFUSED = "transfused"
    EXPIRED = "expired"
    QUARANTINED = "quarantined"

class TemperatureStatus(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"

# SQLAlchemy Models
class BloodUnit(Base):
    __tablename__ = "blood_units"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    blood_group = Column(String(3), nullable=False)
    status = Column(String(20), nullable=False, default=BloodUnitStatus.AVAILABLE.value)
    collection_date = Column(DateTime, nullable=False)
    expiry_date = Column(DateTime, nullable=False)
    volume_ml = Column(Float, nullable=False)
    donor_id = Column(String(50), nullable=True)
    pod_id = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    temperature_logs = relationship("TemperatureLog", back_populates="blood_unit")
    demand_forecasts = relationship("DemandForecast", back_populates="blood_unit")

class TemperatureLog(Base):
    __tablename__ = "temperature_logs"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    blood_unit_id = Column(String(36), ForeignKey("blood_units.id"), nullable=False)
    pod_id = Column(String(50), nullable=False)
    temperature_celsius = Column(Float, nullable=False)
    humidity_percent = Column(Float, nullable=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    status = Column(String(20), nullable=False, default=TemperatureStatus.NORMAL.value)
    
    # Relationships
    blood_unit = relationship("BloodUnit", back_populates="temperature_logs")

class DemandForecast(Base):
    __tablename__ = "demand_forecasts"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    blood_unit_id = Column(String(36), ForeignKey("blood_units.id"), nullable=True)
    blood_group = Column(String(3), nullable=False)
    pod_id = Column(String(50), nullable=False)
    forecast_date = Column(DateTime, nullable=False)
    predicted_demand = Column(Float, nullable=False)
    confidence_interval_lower = Column(Float, nullable=True)
    confidence_interval_upper = Column(Float, nullable=True)
    model_version = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    blood_unit = relationship("BloodUnit", back_populates="demand_forecasts")

class ViabilityPrediction(Base):
    __tablename__ = "viability_predictions"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    blood_unit_id = Column(String(36), ForeignKey("blood_units.id"), nullable=False)
    predicted_expiry_date = Column(DateTime, nullable=False)
    survival_probability = Column(Float, nullable=False)
    risk_score = Column(Float, nullable=False)
    model_version = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

# Pydantic Models for API
class BloodUnitCreate(BaseModel):
    blood_group: BloodGroup
    collection_date: datetime
    volume_ml: float = Field(..., gt=0)
    donor_id: Optional[str] = None
    pod_id: str
    
    @validator('collection_date')
    def validate_collection_date(cls, v):
        if v > datetime.utcnow():
            raise ValueError('Collection date cannot be in the future')
        return v

class BloodUnitResponse(BaseModel):
    id: str
    blood_group: BloodGroup
    status: BloodUnitStatus
    collection_date: datetime
    expiry_date: datetime
    volume_ml: float
    donor_id: Optional[str]
    pod_id: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class TemperatureLogCreate(BaseModel):
    blood_unit_id: str
    pod_id: str
    temperature_celsius: float = Field(..., ge=-50, le=50)
    humidity_percent: Optional[float] = Field(None, ge=0, le=100)
    timestamp: Optional[datetime] = None

class TemperatureLogResponse(BaseModel):
    id: str
    blood_unit_id: str
    pod_id: str
    temperature_celsius: float
    humidity_percent: Optional[float]
    timestamp: datetime
    status: TemperatureStatus
    
    class Config:
        from_attributes = True

class DemandForecastCreate(BaseModel):
    blood_group: BloodGroup
    pod_id: str
    forecast_date: datetime
    predicted_demand: float = Field(..., ge=0)
    confidence_interval_lower: Optional[float] = None
    confidence_interval_upper: Optional[float] = None
    model_version: str

class DemandForecastResponse(BaseModel):
    id: str
    blood_group: BloodGroup
    pod_id: str
    forecast_date: datetime
    predicted_demand: float
    confidence_interval_lower: Optional[float]
    confidence_interval_upper: Optional[float]
    model_version: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class ViabilityPredictionCreate(BaseModel):
    blood_unit_id: str
    predicted_expiry_date: datetime
    survival_probability: float = Field(..., ge=0, le=1)
    risk_score: float = Field(..., ge=0, le=1)
    model_version: str

class ViabilityPredictionResponse(BaseModel):
    id: str
    blood_unit_id: str
    predicted_expiry_date: datetime
    survival_probability: float
    risk_score: float
    model_version: str
    created_at: datetime
    
    class Config:
        from_attributes = True

# ML Data Models
class TimeSeriesDataPoint(BaseModel):
    timestamp: datetime
    value: float
    features: Dict[str, Any] = {}

class SurvivalDataPoint(BaseModel):
    entry_time: datetime
    event_time: Optional[datetime] = None
    is_censored: bool
    features: Dict[str, Any] = {}

class MLPredictionRequest(BaseModel):
    blood_group: Optional[BloodGroup] = None
    pod_id: Optional[str] = None
    forecast_horizon_days: int = Field(7, ge=1, le=30)
    include_confidence_intervals: bool = True

class MLPredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    model_metadata: Dict[str, Any]
    confidence_intervals: Optional[Dict[str, List[float]]] = None
