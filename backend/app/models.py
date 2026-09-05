"""
Database models for MARISA CDSS.
"""

from datetime import datetime

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    JSON,
    Text,
)

from .database import Base


class PredictionLog(Base):
    """
    Stores each MARISA malaria risk prediction and subsequent
    follow-up information.

    Supports:
    - prediction history
    - national monitoring
    - district accountability
    - follow-up tracking
    - drift analysis
    """

    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)

    # --------------------------------------------------
    # Prediction context
    # --------------------------------------------------

    clinic_id = Column(String, nullable=True, index=True)

    district = Column(String, nullable=True, index=True)

    pregnant = Column(Boolean, nullable=False, index=True)

    age = Column(Integer, nullable=False)

    province = Column(String, nullable=True, index=True)

    residence_type = Column(String, nullable=True)

    input_payload = Column(JSON, nullable=False)

    # --------------------------------------------------
    # Model output
    # --------------------------------------------------

    model_version = Column(String, nullable=False)

    risk_score = Column(Float, nullable=False)

    risk_level = Column(String, nullable=False, index=True)

    top_factors = Column(JSON, nullable=True)

    # --------------------------------------------------
    # Follow-up monitoring
    # --------------------------------------------------

    followup_status = Column(
        String,
        nullable=False,
        default="pending",
        index=True,
    )

    tested = Column(Boolean, nullable=True)

    test_result = Column(String, nullable=True)

    referred = Column(Boolean, nullable=True)

    treated = Column(Boolean, nullable=True)

    followup_updated_at = Column(DateTime, nullable=True)

    followup_notes = Column(Text, nullable=True)

    # --------------------------------------------------
    # Timestamp
    # --------------------------------------------------

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
        index=True,
    )