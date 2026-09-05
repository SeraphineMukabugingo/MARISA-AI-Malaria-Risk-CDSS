"""
Pydantic schemas for the MARISA CDSS API.

PatientInput uses simplified, clinician-friendly fields that are
encoded server-side into the exact 66-column feature vector expected
by the trained MARISA Random Forest model.

Additional schemas support:
- prediction explanations
- health monitoring
- model drift monitoring
- patient follow-up
- pregnant women summaries
- national and RBC dashboards
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ==========================================================
# PATIENT INPUT
# ==========================================================

class PatientInput(BaseModel):
    """
    Simplified clinician-facing patient information.

    These fields are converted by ml_model.encode_patient()
    into the exact 66 features required by the trained model.
    """

    clinic_id: Optional[str] = Field(
        None,
        description="Optional clinic identifier for multi-clinic tracking",
    )

    district: Optional[str] = Field(
        None,
        description=(
            "Optional district for district-level monitoring "
            "and RBC reporting"
        ),
    )

    age: int = Field(
        ...,
        ge=15,
        le=49,
        description="Age of the woman in years",
    )

    pregnant: bool = Field(
        ...,
        description="Whether the woman is currently pregnant",
    )

    province: str = Field(
        ...,
        description=(
            "One of: Kigali City, Southern Province, "
            "Northern Province, Eastern Province, "
            "Western Province"
        ),
    )

    residence_type: str = Field(
        ...,
        description="Rural or Urban",
    )

    marital_status: str = Field(
        ...,
        description="never, married, widowed, or divorced",
    )

    education: str = Field(
        ...,
        description="none, primary, secondary, or higher",
    )

    health_insurance: str = Field(
        ...,
        description="yes, no, or unknown",
    )

    has_net: bool = Field(
        ...,
        description="Whether the household has a mosquito bed net",
    )

    itn_treated: Optional[bool] = Field(
        None,
        description=(
            "Whether the mosquito net is insecticide-treated; "
            "only relevant if has_net is true"
        ),
    )

    sprayed_last_12mo: str = Field(
        ...,
        description="yes, no, or unknown",
    )

    water_source: str = Field(
        ...,
        description="improved or unimproved",
    )

    toilet_facility: str = Field(
        ...,
        description="improved or unimproved",
    )

    electricity: bool = Field(
        ...,
        description="Whether the household has electricity",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "clinic_id": "kabuga-hc",
                "district": "Kicukiro",
                "age": 27,
                "pregnant": True,
                "province": "Eastern Province",
                "residence_type": "Rural",
                "marital_status": "married",
                "education": "primary",
                "health_insurance": "no",
                "has_net": True,
                "itn_treated": False,
                "sprayed_last_12mo": "no",
                "water_source": "unimproved",
                "toilet_facility": "unimproved",
                "electricity": False,
            }
        }
    }


# ==========================================================
# MODEL EXPLANATION
# ==========================================================

class FactorContribution(BaseModel):
    """
    One feature contributing to a prediction.
    """

    label: str

    contribution: float = Field(
        ...,
        description="SHAP contribution or equivalent feature contribution",
    )

    direction: str = Field(
        ...,
        description="increases_risk or decreases_risk",
    )


# ==========================================================
# PREDICTION RESPONSE
# ==========================================================

class PredictionResponse(BaseModel):
    """
    Response returned by POST /predict.
    """

    model_config = {
        "protected_namespaces": ()
    }

    risk_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Random Forest model-estimated malaria risk score from 0 to 1. "
            "This score is used for risk stratification and clinical "
            "decision support. It is not a calibrated probability and "
            "does not constitute a malaria diagnosis."
        ),
    )

    risk_level: str = Field(
        ...,
        description="Low / Moderate / High / Very High",
    )

    top_factors: List[FactorContribution] = Field(
        default_factory=list,
        description=(
            "Most influential model features available for the prediction"
        ),
    )

    model_version: str = Field(
        ...,
        description="Version identifier of the deployed MARISA ML model",
    )

    risk_score_note: str = Field(
        ...,
        description=(
            "Important interpretation note explaining how the model "
            "risk score should be used"
        ),
    )

    validation_note: Optional[str] = Field(
        None,
        description=(
            "Optional caution message when pregnant=True, reflecting "
            "the limitations of the pregnant-women validation analysis"
        ),
    )


# ==========================================================
# HEALTH RESPONSE
# ==========================================================

class HealthResponse(BaseModel):
    """
    Response returned by GET /health.
    """

    model_config = {
        "protected_namespaces": ()
    }

    status: str

    model_loaded: bool

    model_version: str


# ==========================================================
# DRIFT SUMMARY RESPONSE
# ==========================================================

class DriftSummaryResponse(BaseModel):
    """
    Aggregate model monitoring statistics.
    """

    total_predictions: int

    predictions_last_7_days: int

    mean_risk_score_overall: Optional[float]

    mean_risk_score_last_7_days: Optional[float]

    high_risk_rate_overall: Optional[float]

    high_risk_rate_last_7_days: Optional[float]

    by_province: Dict[str, Any]

    note: str


# ==========================================================
# FOLLOW-UP UPDATE
# ==========================================================

class FollowupUpdate(BaseModel):
    """
    Information recorded after the initial MARISA prediction.

    This supports monitoring whether a woman identified as high risk
    subsequently received testing, referral, treatment, or other
    documented follow-up.
    """

    tested: bool = Field(
        False,
        description="Whether the patient received malaria testing",
    )

    test_result: Optional[str] = Field(
        None,
        description=(
            "Optional malaria test result, for example "
            "positive or negative"
        ),
    )

    referred: bool = Field(
        False,
        description="Whether the patient was referred for further care",
    )

    treated: bool = Field(
        False,
        description="Whether treatment was recorded",
    )

    notes: Optional[str] = Field(
        None,
        description="Optional follow-up notes",
    )


# ==========================================================
# FOLLOW-UP RESPONSE
# ==========================================================

class FollowupResponse(BaseModel):
    """
    Response returned after updating follow-up information.
    """

    prediction_id: int

    followup_status: str

    message: str


# ==========================================================
# PREGNANT WOMEN SUMMARY
# ==========================================================

class PregnantSummaryResponse(BaseModel):
    """
    National monitoring summary for pregnant women assessed by MARISA.
    """

    total_pregnant_assessed: int

    total_pregnant_high_risk: int

    national_high_risk_rate: Optional[float]

    national_followup_completion_rate: Optional[float]

    by_province: Dict[str, Any]

    by_district: Dict[str, Any]

    flagged_for_followup: List[Dict[str, Any]]

    note: str