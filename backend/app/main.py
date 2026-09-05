"""
MARISA CDSS Prediction API.

Endpoints:

    POST  /predict
        Predict malaria risk and provide an explainable AI-based result.

    GET   /health
        Check API and model status.

    GET   /drift-summary
        Monitor aggregate prediction statistics and potential model drift.

    PATCH /followup/{prediction_id}
        Record follow-up actions after a prediction.

    GET   /pregnant-summary
        Summarize malaria risk and follow-up status among pregnant women.

    GET   /dashboard
        Provide an overall national dashboard summary.

    GET   /daily-trend
        Provide daily prediction and high-risk trends.

    GET   /rbc-dashboard
        Provide district-level accountability and follow-up monitoring.

Run locally:

    uvicorn app.main:app --reload --port 8000

API documentation:

    http://localhost:8000/docs
"""


# ==========================================================
# IMPORTS
# ==========================================================

from datetime import datetime, timedelta

import numpy as np
import shap

from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
)

from fastapi.middleware.cors import CORSMiddleware

from sqlalchemy import func
from sqlalchemy.orm import Session


from . import ml_model

from .database import (
    Base,
    engine,
    get_db,
)

from .models import PredictionLog

from .schemas import (
    PatientInput,
    PredictionResponse,
    FactorContribution,
    HealthResponse,
    DriftSummaryResponse,
    FollowupUpdate,
    FollowupResponse,
    PregnantSummaryResponse,
)


# ==========================================================
# FASTAPI APPLICATION
# ==========================================================

app = FastAPI(

    title="MARISA CDSS API",

    description=(
        "Explainable AI-powered malaria risk prediction and "
        "clinical decision support system for women of "
        "reproductive age in Rwanda"
    ),

    version="1.0.0",

)


# ==========================================================
# CORS
# ==========================================================

app.add_middleware(

    CORSMiddleware,

    # Development configuration.
    # Before production deployment, restrict this to
    # the actual frontend domain(s).

    allow_origins=[
        "*",
    ],

    allow_methods=[
        "*",
    ],

    allow_headers=[
        "*",
    ],

)


# ==========================================================
# DATABASE INITIALIZATION
# ==========================================================

Base.metadata.create_all(
    bind=engine
)


# ==========================================================
# LOAD ML MODEL
# ==========================================================

model = ml_model.load_model()


# ==========================================================
# EXACT MODEL FEATURE SCHEMA
# ==========================================================

FEATURE_COLUMNS = list(
    ml_model.DEFAULT_FEATURE_COLUMNS
)


# ==========================================================
# VALIDATE MODEL FEATURE SCHEMA
# ==========================================================

if hasattr(
    model,
    "feature_names_in_",
):

    model_feature_columns = list(
        model.feature_names_in_
    )


    if model_feature_columns != FEATURE_COLUMNS:

        raise RuntimeError(

            "MARISA model feature schema does not match "
            "DEFAULT_FEATURE_COLUMNS."

        )


if len(
    FEATURE_COLUMNS
) != 66:

    raise RuntimeError(

        "MARISA feature schema is invalid. "
        f"Expected 66 features but found "
        f"{len(FEATURE_COLUMNS)}."

    )


print(
    "MARISA feature schema validated:",
    len(FEATURE_COLUMNS),
    "features",
)


# ==========================================================
# SHAP EXPLAINER
# ==========================================================

explainer = None


try:

    explainer = shap.TreeExplainer(
        model
    )


    print(
        "MARISA SHAP TreeExplainer initialized successfully."
    )


except Exception as e:

    print(

        "WARNING: SHAP explainer could not be initialized:",

        e,

    )


    explainer = None


# ==========================================================
# SHAP DISPLAY HELPERS
# ==========================================================

def base_variable_group(
    feature: str,
) -> str:
    """
    Group related one-hot encoded features so that one
    real-world variable is not displayed multiple times.
    """

    if feature.startswith(
        "health_insurance_coverage__"
    ):

        return "health_insurance"


    if feature.startswith(
        "insecticidetreated_net_itn__"
    ):

        return "itn_status"


    if feature.startswith(
        "has_dwelling_been_sprayed_against_mosquitoes_in_last_12_months__"
    ):

        return "sprayed_status"


    if feature.startswith(
        "current_marital_status__"
    ):

        return "marital_status"


    if feature.startswith(
        "source_of_drinking_water__"
    ):

        return "water_source"


    if feature.startswith(
        "type_of_toilet_facility__"
    ):

        return "toilet_facility"


    if feature.startswith(
        "province_name_"
    ):

        return "province"


    if feature.startswith(
        "population_density_category_"
    ):

        return "population_density_category"


    if feature.startswith(
        "residence_type_"
    ):

        return "residence_type"


    if feature.startswith(
        "type_of_place_of_residence__"
    ):

        return "residence_type"


    return feature


# ==========================================================
# CONTINUOUS FEATURES
# ==========================================================

CONTINUOUS_FEATURES = {

    "age_of_household_members__hv105",

    "elevation",

    "annual_humidity",

    "annual_rainfall",

    "annual_temperature",

    "population_density",

    "nearest_health_facility_km_fixed",

    "health_accessibility_score",

    "wealth_index_combined__hv270",

    "highest_educational_level__v106",

}


# ==========================================================
# FEATURE LABEL HELPER
# ==========================================================

def contextual_feature_label(
    feature: str,
    patient_value: float,
) -> str:
    """
    Produce a clinician-readable feature label.

    For one-hot encoded variables, indicate whether the
    category applies to the current patient.
    """

    label = ml_model.friendly_label(
        feature
    )


    if feature in CONTINUOUS_FEATURES:

        return (

            f"{label}: "
            f"{float(patient_value):g}"

        )


    applies = (

        "Yes"

        if float(patient_value) >= 0.5

        else "No"

    )


    return (

        f"{label} "
        f"(Applies: {applies})"

    )


# ==========================================================
# FOLLOW-UP STATUS HELPER
# ==========================================================

def calculate_followup_status(
    log: PredictionLog,
) -> str:
    """
    Determine follow-up completion status.

    Completed means at least one meaningful clinical
    follow-up action has been recorded.
    """

    if (

        log.tested

        or log.referred

        or log.treated

    ):

        return "completed"


    return "pending"


# ==========================================================
# HIGH-RISK HELPER
# ==========================================================

def is_high_risk(
    risk_level: str,
) -> bool:
    """
    Return True if prediction is classified as High
    or Very High risk.
    """

    return risk_level in [

        "High",

        "Very High",

    ]


# ==========================================================
# HEALTH ENDPOINT
# ==========================================================

@app.get(

    "/health",

    response_model=HealthResponse,

)

def health():
    """
    Check whether the MARISA API and trained model
    are available.
    """

    return HealthResponse(

        status="ok",

        model_loaded=(
            model is not None
        ),

        model_version=(

            ml_model.MODEL_VERSION

            + (

                " [PLACEHOLDER]"

                if ml_model.is_placeholder()

                else ""

            )

        ),

    )


# ==========================================================
# PREDICTION ENDPOINT
# ==========================================================

@app.post(

    "/predict",

    response_model=PredictionResponse,

)

def predict(

    patient: PatientInput,

    db: Session = Depends(get_db),

):
    """
    Generate a malaria risk prediction.

    The result is a model-based malaria risk score.

    It does not constitute a malaria diagnosis and must
    not replace standard malaria testing procedures.
    """


    # ------------------------------------------------------
    # ENCODE PATIENT
    # ------------------------------------------------------

    X = ml_model.encode_patient(
        patient
    )


    # ------------------------------------------------------
    # VALIDATE FEATURE SCHEMA
    # ------------------------------------------------------

    if list(
        X.columns
    ) != FEATURE_COLUMNS:

        raise HTTPException(

            status_code=500,

            detail=(

                "Internal feature encoding error: encoded "
                "feature columns do not match the MARISA "
                "model feature schema."

            ),

        )


    if X.shape[1] != 66:

        raise HTTPException(

            status_code=500,

            detail=(

                "Internal feature encoding error: MARISA "
                f"expects 66 features but received "
                f"{X.shape[1]}."

            ),

        )


    # ------------------------------------------------------
    # GENERATE MALARIA RISK SCORE
    # ------------------------------------------------------

    try:

        probabilities = model.predict_proba(
            X
        )


        proba = float(
            probabilities[0][1]
        )


    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=(

                "MARISA model prediction failed: "
                f"{e}"

            ),

        )


    # ------------------------------------------------------
    # CLASSIFY RISK
    # ------------------------------------------------------

    risk_level = ml_model.classify_risk(
        proba
    )


    # ------------------------------------------------------
    # SHAP EXPLANATION
    # ------------------------------------------------------

    top_factors = []


    try:

        if explainer is not None:


            shap_values = explainer.shap_values(
                X
            )


            # --------------------------------------------------
            # HANDLE DIFFERENT SHAP OUTPUT FORMATS
            # --------------------------------------------------

            if isinstance(
                shap_values,
                list,
            ):


                # Older SHAP format:
                #
                # [
                #   class_0_values,
                #   class_1_values
                # ]
                #
                # Select malaria-positive class.

                if len(
                    shap_values
                ) > 1:

                    sv = np.asarray(
                        shap_values[1]
                    )


                else:

                    sv = np.asarray(
                        shap_values[0]
                    )


                # Remove sample dimension.

                if sv.ndim >= 2:

                    sv = sv[0]


            else:


                shap_array = np.asarray(
                    shap_values
                )


                # Modern SHAP format:
                #
                # (samples, features, classes)

                if shap_array.ndim == 3:

                    sv = shap_array[
                        0,
                        :,
                        1,
                    ]


                # (samples, features)

                elif shap_array.ndim == 2:

                    sv = shap_array[
                        0
                    ]


                # (features,)

                elif shap_array.ndim == 1:

                    sv = shap_array


                else:

                    raise ValueError(

                        "Unexpected SHAP output shape: "
                        f"{shap_array.shape}"

                    )


            # --------------------------------------------------
            # ENSURE ONE-DIMENSIONAL CONTRIBUTIONS
            # --------------------------------------------------

            sv = np.asarray(
                sv
            ).reshape(
                -1
            )


            # --------------------------------------------------
            # VALIDATE SHAP FEATURE COUNT
            # --------------------------------------------------

            if len(
                sv
            ) != len(
                FEATURE_COLUMNS
            ):

                raise ValueError(

                    "SHAP feature count does not match "
                    "MARISA model feature schema. "
                    f"SHAP features={len(sv)}, "
                    f"expected features="
                    f"{len(FEATURE_COLUMNS)}"

                )


            # --------------------------------------------------
            # BUILD FEATURE RECORDS
            # --------------------------------------------------

            feature_records = []


            for index, (

                feature,

                contribution,

            ) in enumerate(

                zip(
                    FEATURE_COLUMNS,
                    sv,
                )

            ):


                patient_value = float(

                    X.iloc[
                        0,
                        index,
                    ]

                )


                feature_records.append(

                    {

                        "feature": feature,

                        "patient_value": (
                            patient_value
                        ),

                        "contribution": (

                            float(
                                contribution
                            )

                        ),

                        "group": (

                            base_variable_group(
                                feature
                            )

                        ),

                    }

                )


            # --------------------------------------------------
            # SELECT ONE FEATURE PER REAL-WORLD VARIABLE
            # --------------------------------------------------
            #
            # Example:
            #
            # health_insurance_coverage__sh11_0.0
            # health_insurance_coverage__sh11_1.0
            #
            # should not both appear as separate top factors.
            #
            # Prefer categories that actually apply to
            # the current patient.
            # --------------------------------------------------

            best_by_group = {}


            for item in feature_records:


                group = item[
                    "group"
                ]


                if group not in best_by_group:

                    best_by_group[
                        group
                    ] = item

                    continue


                current = best_by_group[
                    group
                ]


                current_active = (

                    abs(

                        current[
                            "patient_value"
                        ]

                    ) >= 0.5

                )


                new_active = (

                    abs(

                        item[
                            "patient_value"
                        ]

                    ) >= 0.5

                )


                # Prefer active category.

                if (

                    new_active

                    and not current_active

                ):

                    best_by_group[
                        group
                    ] = item


                # If both have same active status,
                # choose strongest SHAP contribution.

                elif (

                    new_active
                    == current_active

                    and

                    abs(

                        item[
                            "contribution"
                        ]

                    )

                    >

                    abs(

                        current[
                            "contribution"
                        ]

                    )

                ):

                    best_by_group[
                        group
                    ] = item


            # --------------------------------------------------
            # SORT BY ABSOLUTE CONTRIBUTION
            # --------------------------------------------------

            strongest_features = sorted(

                best_by_group.values(),

                key=lambda item:

                    abs(

                        item[
                            "contribution"
                        ]

                    ),

                reverse=True,

            )[:6]


            # --------------------------------------------------
            # BUILD CLINICIAN-FACING EXPLANATIONS
            # --------------------------------------------------

            top_factors = [

                FactorContribution(

                    label=(

                        contextual_feature_label(

                            item[
                                "feature"
                            ],

                            item[
                                "patient_value"
                            ],

                        )

                    ),

                    contribution=(

                        round(

                            item[
                                "contribution"
                            ],

                            4,

                        )

                    ),

                    direction=(

                        "increases_risk"

                        if item[
                            "contribution"
                        ] > 0

                        else

                        "decreases_risk"

                    ),

                )

                for item

                in strongest_features

            ]


    except Exception as e:


        print(

            "WARNING: SHAP explanation failed:",

            repr(
                e
            ),

        )


        # Prediction must still succeed if SHAP fails.

        top_factors = []


    # ------------------------------------------------------
    # PREGNANT WOMEN VALIDATION WARNING
    # ------------------------------------------------------

    validation_note = None


    if patient.pregnant:


        validation_note = (

            "This model's validation specifically among pregnant women "
            "was based on only 23 confirmed cases and showed unstable "
            "cross-validated performance (mean recall 11%). "
            "Always follow standard malaria testing protocol regardless "
            "of this score."

        )


    # ------------------------------------------------------
    # SAVE PREDICTION
    # ------------------------------------------------------

    log = PredictionLog(

        clinic_id=patient.clinic_id,

        district=patient.district,

        pregnant=patient.pregnant,

        age=patient.age,

        province=patient.province,

        residence_type=patient.residence_type,

        input_payload=patient.model_dump(),

        model_version=(
            ml_model.MODEL_VERSION
        ),

        risk_score=proba,

        risk_level=risk_level,

        top_factors=[

            factor.model_dump()

            for factor

            in top_factors

        ],

    )


    try:


        db.add(
            log
        )


        db.commit()


        db.refresh(
            log
        )


    except Exception as e:


        db.rollback()


        raise HTTPException(

            status_code=500,

            detail=(

                "Prediction was generated but could not "
                "be saved to the database: "
                f"{e}"

            ),

        )


    # ------------------------------------------------------
    # RETURN RESPONSE
    # ------------------------------------------------------

    return PredictionResponse(

        risk_score=round(

            proba,

            4,

        ),

        risk_level=risk_level,

        top_factors=top_factors,

        model_version=(

            ml_model.MODEL_VERSION

            + (

                " [PLACEHOLDER]"

                if ml_model.is_placeholder()

                else ""

            )

        ),

        risk_score_note=(

            "This is a Random Forest model risk score used for "
            "risk stratification and clinical decision support. "
            "It is not an isotonic-calibrated probability and "
            "should not be interpreted as an exact probability "
            "of malaria positivity or as a diagnosis."

        ),

        validation_note=validation_note,

    )


# ==========================================================
# DRIFT SUMMARY
# ==========================================================

@app.get(

    "/drift-summary",

    response_model=DriftSummaryResponse,

)

def drift_summary(

    db: Session = Depends(get_db),

):


    # ------------------------------------------------------
    # OVERALL PREDICTION COUNT
    # ------------------------------------------------------

    total = (

        db.query(

            func.count(
                PredictionLog.id
            )

        ).scalar()

        or 0

    )


    # ------------------------------------------------------
    # LAST 7 DAYS
    # ------------------------------------------------------

    cutoff = (

        datetime.utcnow()

        - timedelta(
            days=7
        )

    )


    recent_query = (

        db.query(
            PredictionLog
        )

        .filter(

            PredictionLog.created_at
            >= cutoff

        )

    )


    recent_count = (
        recent_query.count()
    )


    # ------------------------------------------------------
    # MEAN RISK SCORE
    # ------------------------------------------------------

    mean_overall = (

        db.query(

            func.avg(
                PredictionLog.risk_score
            )

        ).scalar()

    )


    mean_recent = (

        recent_query

        .with_entities(

            func.avg(
                PredictionLog.risk_score
            )

        )

        .scalar()

    )


    # ------------------------------------------------------
    # HIGH-RISK COUNTS
    # ------------------------------------------------------

    high_overall = (

        db.query(

            func.count(
                PredictionLog.id
            )

        )

        .filter(

            PredictionLog.risk_level.in_(

                [

                    "High",

                    "Very High",

                ]

            )

        )

        .scalar()

        or 0

    )


    high_recent = (

        recent_query

        .filter(

            PredictionLog.risk_level.in_(

                [

                    "High",

                    "Very High",

                ]

            )

        )

        .count()

    )


    # ------------------------------------------------------
    # PROVINCE STATISTICS
    # ------------------------------------------------------

    by_province = {}


    rows = (

        db.query(

            PredictionLog.province,

            func.count(
                PredictionLog.id
            ),

            func.avg(
                PredictionLog.risk_score
            ),

        )

        .group_by(
            PredictionLog.province
        )

        .all()

    )


    for province, count, mean_score in rows:


        by_province[
            province or "unknown"
        ] = {

            "count": count,

            "mean_risk_score": (

                round(
                    mean_score,
                    4,
                )

                if mean_score is not None

                else None

            ),

        }


    return DriftSummaryResponse(

        total_predictions=total,

        predictions_last_7_days=recent_count,

        mean_risk_score_overall=(

            round(
                mean_overall,
                4,
            )

            if mean_overall is not None

            else None

        ),

        mean_risk_score_last_7_days=(

            round(
                mean_recent,
                4,
            )

            if mean_recent is not None

            else None

        ),

        high_risk_rate_overall=(

            round(

                high_overall
                / total,

                4,

            )

            if total

            else None

        ),

        high_risk_rate_last_7_days=(

            round(

                high_recent
                / recent_count,

                4,

            )

            if recent_count

            else None

        ),

        by_province=by_province,

        note=(

            "Aggregate monitoring summary for MARISA prediction activity. "
            "Changes in prediction rates may indicate changes in the "
            "population being assessed or possible model-data drift."

        ),

    )


# ==========================================================
# FOLLOW-UP UPDATE
# ==========================================================

@app.patch(

    "/followup/{prediction_id}",

    response_model=FollowupResponse,

)

def update_followup(

    prediction_id: int,

    update: FollowupUpdate,

    db: Session = Depends(get_db),

):


    prediction = (

        db.query(
            PredictionLog
        )

        .filter(

            PredictionLog.id
            == prediction_id

        )

        .first()

    )


    if prediction is None:


        raise HTTPException(

            status_code=404,

            detail=(
                "Prediction record not found"
            ),

        )


    # ------------------------------------------------------
    # UPDATE FOLLOW-UP INFORMATION
    # ------------------------------------------------------

    prediction.tested = (
        update.tested
    )


    prediction.test_result = (
        update.test_result
    )


    prediction.referred = (
        update.referred
    )


    prediction.treated = (
        update.treated
    )


    prediction.followup_notes = (
        update.notes
    )


    # ------------------------------------------------------
    # DETERMINE FOLLOW-UP STATUS
    # ------------------------------------------------------

    prediction.followup_status = (

        calculate_followup_status(
            prediction
        )

    )


    prediction.followup_updated_at = (
        datetime.utcnow()
    )


    try:


        db.commit()


        db.refresh(
            prediction
        )


    except Exception as e:


        db.rollback()


        raise HTTPException(

            status_code=500,

            detail=(

                "Could not update follow-up information: "
                f"{e}"

            ),

        )


    return FollowupResponse(

        prediction_id=prediction.id,

        followup_status=(
            prediction.followup_status
        ),

        message=(

            "Follow-up information updated successfully"

        ),

    )


# ==========================================================
# PREGNANT WOMEN SUMMARY
# ==========================================================

@app.get(

    "/pregnant-summary",

    response_model=PregnantSummaryResponse,

)

def pregnant_summary(

    db: Session = Depends(get_db),

):


    pregnant_records = (

        db.query(
            PredictionLog
        )

        .filter(
            PredictionLog.pregnant.is_(True)
        )

    )


    total_pregnant = (
        pregnant_records.count()
    )


    high_risk_pregnant = (

        pregnant_records

        .filter(

            PredictionLog.risk_level.in_(

                [

                    "High",

                    "Very High",

                ]

            )

        )

        .count()

    )


    completed_followups = (

        pregnant_records

        .filter(

            PredictionLog.followup_status
            == "completed"

        )

        .count()

    )


    # ------------------------------------------------------
    # PROVINCE STATISTICS
    # ------------------------------------------------------

    by_province = {}


    province_rows = (

        db.query(

            PredictionLog.province,

            func.count(
                PredictionLog.id
            ),

            func.avg(
                PredictionLog.risk_score
            ),

        )

        .filter(
            PredictionLog.pregnant.is_(True)
        )

        .group_by(
            PredictionLog.province
        )

        .all()

    )


    for province, count, mean_score in province_rows:


        by_province[
            province or "unknown"
        ] = {

            "count": count,

            "mean_risk_score": (

                round(
                    mean_score,
                    4,
                )

                if mean_score is not None

                else None

            ),

        }


    # ------------------------------------------------------
    # DISTRICT STATISTICS
    # ------------------------------------------------------

    by_district = {}


    district_rows = (

        db.query(

            PredictionLog.district,

            func.count(
                PredictionLog.id
            ),

            func.avg(
                PredictionLog.risk_score
            ),

        )

        .filter(
            PredictionLog.pregnant.is_(True)
        )

        .group_by(
            PredictionLog.district
        )

        .all()

    )


    for district, count, mean_score in district_rows:


        by_district[
            district or "unknown"
        ] = {

            "count": count,

            "mean_risk_score": (

                round(
                    mean_score,
                    4,
                )

                if mean_score is not None

                else None

            ),

        }


    # ------------------------------------------------------
    # FLAGGED CASES FOR FOLLOW-UP
    # ------------------------------------------------------

    flagged_records = (

        pregnant_records

        .filter(

            PredictionLog.risk_level.in_(

                [

                    "High",

                    "Very High",

                ]

            )

        )

        .filter(

            PredictionLog.followup_status
            != "completed"

        )

        .order_by(

            PredictionLog.created_at.desc()

        )

        .limit(
            50
        )

        .all()

    )


    flagged_for_followup = []


    for record in flagged_records:


        flagged_for_followup.append(

            {

                "prediction_id": record.id,

                "clinic_id": record.clinic_id,

                "district": record.district,

                "province": record.province,

                "age": record.age,

                "risk_score": (

                    round(
                        record.risk_score,
                        4,
                    )

                ),

                "risk_level": record.risk_level,

                "followup_status": (
                    record.followup_status
                ),

                "created_at": (

                    record.created_at.isoformat()

                    if record.created_at

                    else None

                ),

            }

        )


    return PregnantSummaryResponse(

        total_pregnant_assessed=(
            total_pregnant
        ),

        total_pregnant_high_risk=(
            high_risk_pregnant
        ),

        national_high_risk_rate=(

            round(

                high_risk_pregnant
                / total_pregnant,

                4,

            )

            if total_pregnant

            else None

        ),

        national_followup_completion_rate=(

            round(

                completed_followups
                / total_pregnant,

                4,

            )

            if total_pregnant

            else None

        ),

        by_province=by_province,

        by_district=by_district,

        flagged_for_followup=(
            flagged_for_followup
        ),

        note=(

            "Summary of pregnant women assessed by MARISA. "
            "High-risk records without completed follow-up should "
            "be reviewed by the responsible health facility."

        ),

    )


# ==========================================================
# NATIONAL DASHBOARD
# ==========================================================

@app.get(
    "/dashboard",
)

def dashboard(

    db: Session = Depends(get_db),

):


    total_predictions = (

        db.query(

            func.count(
                PredictionLog.id
            )

        ).scalar()

        or 0

    )


    total_pregnant = (

        db.query(

            func.count(
                PredictionLog.id
            )

        )

        .filter(
            PredictionLog.pregnant.is_(True)
        )

        .scalar()

        or 0

    )


    total_high_risk = (

        db.query(

            func.count(
                PredictionLog.id
            )

        )

        .filter(

            PredictionLog.risk_level.in_(

                [

                    "High",

                    "Very High",

                ]

            )

        )

        .scalar()

        or 0

    )


    pregnant_high_risk = (

        db.query(

            func.count(
                PredictionLog.id
            )

        )

        .filter(
            PredictionLog.pregnant.is_(True)
        )

        .filter(

            PredictionLog.risk_level.in_(

                [

                    "High",

                    "Very High",

                ]

            )

        )

        .scalar()

        or 0

    )


    followup_completed = (

        db.query(

            func.count(
                PredictionLog.id
            )

        )

        .filter(

            PredictionLog.followup_status
            == "completed"

        )

        .scalar()

        or 0

    )


    mean_risk_score = (

        db.query(

            func.avg(
                PredictionLog.risk_score
            )

        ).scalar()

    )


    # ------------------------------------------------------
    # PROVINCE BREAKDOWN
    # ------------------------------------------------------

    province_breakdown = {}


    rows = (

        db.query(

            PredictionLog.province,

            func.count(
                PredictionLog.id
            ),

            func.avg(
                PredictionLog.risk_score
            ),

        )

        .group_by(
            PredictionLog.province
        )

        .all()

    )


    for province, count, mean_score in rows:


        province_breakdown[
            province or "unknown"
        ] = {

            "predictions": count,

            "mean_risk_score": (

                round(
                    mean_score,
                    4,
                )

                if mean_score is not None

                else None

            ),

        }


    return {

        "total_predictions": (
            total_predictions
        ),

        "total_pregnant_assessed": (
            total_pregnant
        ),

        "total_high_risk": (
            total_high_risk
        ),

        "total_pregnant_high_risk": (
            pregnant_high_risk
        ),

        "overall_high_risk_rate": (

            round(

                total_high_risk
                / total_predictions,

                4,

            )

            if total_predictions

            else None

        ),

        "followup_completed": (
            followup_completed
        ),

        "followup_completion_rate": (

            round(

                followup_completed
                / total_predictions,

                4,

            )

            if total_predictions

            else None

        ),

        "mean_risk_score": (

            round(
                mean_risk_score,
                4,
            )

            if mean_risk_score is not None

            else None

        ),

        "by_province": (
            province_breakdown
        ),

        "generated_at": (
            datetime.utcnow().isoformat()
        ),

    }


# ==========================================================
# DAILY TREND
# ==========================================================

@app.get(
    "/daily-trend",
)

def daily_trend(

    db: Session = Depends(get_db),

):


    rows = (

        db.query(

            func.date(
                PredictionLog.created_at
            ).label(
                "prediction_date"
            ),

            func.count(
                PredictionLog.id
            ).label(
                "total_predictions"
            ),

        )

        .group_by(

            func.date(
                PredictionLog.created_at
            )

        )

        .order_by(

            func.date(
                PredictionLog.created_at
            )

        )

        .all()

    )


    trend = []


    for row in rows:


        date_value = row.prediction_date


        high_risk_count = (

            db.query(

                func.count(
                    PredictionLog.id
                )

            )

            .filter(

                func.date(
                    PredictionLog.created_at
                )
                == date_value

            )

            .filter(

                PredictionLog.risk_level.in_(

                    [

                        "High",

                        "Very High",

                    ]

                )

            )

            .scalar()

            or 0

        )


        pregnant_count = (

            db.query(

                func.count(
                    PredictionLog.id
                )

            )

            .filter(

                func.date(
                    PredictionLog.created_at
                )
                == date_value

            )

            .filter(
                PredictionLog.pregnant.is_(True)
            )

            .scalar()

            or 0

        )


        trend.append(

            {

                "date": str(
                    date_value
                ),

                "total_predictions": (
                    row.total_predictions
                ),

                "high_risk_predictions": (
                    high_risk_count
                ),

                "pregnant_predictions": (
                    pregnant_count
                ),

                "high_risk_rate": (

                    round(

                        high_risk_count
                        / row.total_predictions,

                        4,

                    )

                    if row.total_predictions

                    else 0

                ),

            }

        )


    return {

        "total_days": len(
            trend
        ),

        "daily_trend": trend,

    }


# ==========================================================
# RBC DASHBOARD
# ==========================================================

@app.get(
    "/rbc-dashboard",
)

def rbc_dashboard(

    db: Session = Depends(get_db),

):


    # ------------------------------------------------------
    # DISTRICT AGGREGATION
    # ------------------------------------------------------

    district_rows = (

        db.query(
            PredictionLog.district
        )

        .distinct()

        .all()

    )


    districts = []


    for (
        district,
    ) in district_rows:


        district_name = (

            district
            or "unknown"

        )


        if district is None:

            base_query = (

                db.query(
                    PredictionLog
                )

                .filter(
                    PredictionLog.district.is_(None)
                )

            )

        else:

            base_query = (

                db.query(
                    PredictionLog
                )

                .filter(

                    PredictionLog.district
                    == district

                )

            )


        total = (
            base_query.count()
        )


        pregnant = (

            base_query

            .filter(
                PredictionLog.pregnant.is_(True)
            )

            .count()

        )


        high_risk = (

            base_query

            .filter(

                PredictionLog.risk_level.in_(

                    [

                        "High",

                        "Very High",

                    ]

                )

            )

            .count()

        )


        completed_followup = (

            base_query

            .filter(

                PredictionLog.followup_status
                == "completed"

            )

            .count()

        )


        districts.append(

            {

                "district": (
                    district_name
                ),

                "total_predictions": (
                    total
                ),

                "pregnant_assessed": (
                    pregnant
                ),

                "high_risk_cases": (
                    high_risk
                ),

                "high_risk_rate": (

                    round(

                        high_risk
                        / total,

                        4,

                    )

                    if total

                    else 0

                ),

                "followup_completed": (
                    completed_followup
                ),

                "followup_completion_rate": (

                    round(

                        completed_followup
                        / total,

                        4,

                    )

                    if total

                    else 0

                ),

            }

        )


    # ------------------------------------------------------
    # LIVE FLAGGED CASES
    # ------------------------------------------------------

    flagged_cases = (

        db.query(
            PredictionLog
        )

        .filter(
            PredictionLog.pregnant.is_(True)
        )

        .filter(

            PredictionLog.risk_level.in_(

                [

                    "High",

                    "Very High",

                ]

            )

        )

        .filter(

            PredictionLog.followup_status
            != "completed"

        )

        .order_by(

            PredictionLog.created_at.desc()

        )

        .limit(
            100
        )

        .all()

    )


    flagged = []


    for record in flagged_cases:


        flagged.append(

            {

                "prediction_id": (
                    record.id
                ),

                "district": (
                    record.district
                ),

                "province": (
                    record.province
                ),

                "clinic_id": (
                    record.clinic_id
                ),

                "age": (
                    record.age
                ),

                "risk_score": (

                    round(

                        record.risk_score,

                        4,

                    )

                ),

                "risk_level": (
                    record.risk_level
                ),

                "followup_status": (
                    record.followup_status
                ),

                "tested": (
                    record.tested
                ),

                "test_result": (
                    record.test_result
                ),

                "referred": (
                    record.referred
                ),

                "treated": (
                    record.treated
                ),

                "created_at": (

                    record.created_at.isoformat()

                    if record.created_at

                    else None

                ),

            }

        )


    return {

        "dashboard_type": (

            "Rwanda Biomedical Centre "
            "District Accountability Dashboard"

        ),

        "districts": (
            districts
        ),

        "live_flagged_cases": (
            flagged
        ),

        "generated_at": (
            datetime.utcnow().isoformat()
        ),

        "note": (

            "District-level monitoring of MARISA predictions, "
            "high-risk cases, and documented follow-up actions."

        ),

    }