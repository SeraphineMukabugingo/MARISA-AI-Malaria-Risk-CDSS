"""
MARISA
Explainable AI-Powered Malaria Risk Prediction System
=====================================================

Model loading, feature encoding, risk classification, and
clinician-readable feature explanations for the MARISA backend.

PRODUCTION MODEL
----------------

This module loads the final trained Random Forest model:

    backend/app/artifacts/rf_final_model.pkl

The model expects exactly 66 features.

IMPORTANT
---------

The deployed model is the final RandomForestClassifier exported
from the MARISA modeling pipeline.

This module does NOT use:

    rf_calibrated.pkl

The returned model score is therefore a Random Forest risk score,
not an isotonic-calibrated probability.
"""

import json
import os
from typing import Dict, List, Optional

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


# ==========================================================
# PATH CONFIGURATION
# ==========================================================

ARTIFACT_DIR = os.path.join(
    os.path.dirname(__file__),
    "artifacts",
)

MODEL_PATH = os.path.join(
    ARTIFACT_DIR,
    "rf_final_model.pkl",
)

COLUMNS_PATH = os.path.join(
    ARTIFACT_DIR,
    "feature_columns.json",
)


# ==========================================================
# MODEL VERSION
# ==========================================================

MODEL_VERSION = "marisa-rf-final-v1"


# ==========================================================
# MODEL STATE
# ==========================================================

_model = None
_feature_columns = None
_is_placeholder = False


# ==========================================================
# DEFAULT FEATURE SCHEMA
#
# Exact 66 features used by rf_final_model.pkl.
#
# Their order must match the model training order.
# ==========================================================

DEFAULT_FEATURE_COLUMNS = [

    # ------------------------------------------------------
    # NUMERIC / BINARY FEATURES
    # ------------------------------------------------------

    "has_mosquito_bed_net_for_sleeping__hv227",
    "has_electricity__hv206",
    "currently_pregnant__sb217",
    "highest_educational_level__v106",
    "wealth_index_combined__hv270",
    "age_of_household_members__hv105",
    "elevation",
    "annual_humidity",
    "annual_rainfall",
    "annual_temperature",
    "population_density",
    "nearest_health_facility_km_fixed",
    "health_accessibility_score",

    # ------------------------------------------------------
    # PROVINCE
    # ------------------------------------------------------

    "province_name_Eastern Province",
    "province_name_Kigali City",
    "province_name_Northern Province",
    "province_name_Southern Province",
    "province_name_Western Province",

    # ------------------------------------------------------
    # RESIDENCE TYPE
    # ------------------------------------------------------

    "residence_type_Rural",
    "residence_type_Urban",

    # ------------------------------------------------------
    # POPULATION DENSITY CATEGORY
    # ------------------------------------------------------

    "population_density_category_High Density",
    "population_density_category_Low Density",
    "population_density_category_Medium Density",
    "population_density_category_Very High Density",

    # ------------------------------------------------------
    # DRINKING WATER SOURCE
    # ------------------------------------------------------

    "source_of_drinking_water__hv201_11.0",
    "source_of_drinking_water__hv201_12.0",
    "source_of_drinking_water__hv201_13.0",
    "source_of_drinking_water__hv201_14.0",
    "source_of_drinking_water__hv201_21.0",
    "source_of_drinking_water__hv201_31.0",
    "source_of_drinking_water__hv201_32.0",
    "source_of_drinking_water__hv201_41.0",
    "source_of_drinking_water__hv201_42.0",
    "source_of_drinking_water__hv201_43.0",
    "source_of_drinking_water__hv201_51.0",
    "source_of_drinking_water__hv201_61.0",
    "source_of_drinking_water__hv201_62.0",
    "source_of_drinking_water__hv201_71.0",
    "source_of_drinking_water__hv201_96.0",

    # ------------------------------------------------------
    # TOILET FACILITY
    # ------------------------------------------------------

    "type_of_toilet_facility__hv205_11.0",
    "type_of_toilet_facility__hv205_12.0",
    "type_of_toilet_facility__hv205_13.0",
    "type_of_toilet_facility__hv205_14.0",
    "type_of_toilet_facility__hv205_21.0",
    "type_of_toilet_facility__hv205_22.0",
    "type_of_toilet_facility__hv205_23.0",
    "type_of_toilet_facility__hv205_31.0",
    "type_of_toilet_facility__hv205_41.0",
    "type_of_toilet_facility__hv205_96.0",

    # ------------------------------------------------------
    # MARITAL STATUS
    # ------------------------------------------------------

    "current_marital_status__hv115_0.0",
    "current_marital_status__hv115_1.0",
    "current_marital_status__hv115_2.0",
    "current_marital_status__hv115_3.0",
    "current_marital_status__hv115_4.0",
    "current_marital_status__hv115_5.0",

    # ------------------------------------------------------
    # HEALTH INSURANCE
    # ------------------------------------------------------

    "health_insurance_coverage__sh11_0.0",
    "health_insurance_coverage__sh11_1.0",
    "health_insurance_coverage__sh11_8.0",

    # ------------------------------------------------------
    # DWELLING SPRAYING
    # ------------------------------------------------------

    (
        "has_dwelling_been_sprayed_against_mosquitoes_"
        "in_last_12_months__hv253_0.0"
    ),

    (
        "has_dwelling_been_sprayed_against_mosquitoes_"
        "in_last_12_months__hv253_1.0"
    ),

    (
        "has_dwelling_been_sprayed_against_mosquitoes_"
        "in_last_12_months__hv253_8.0"
    ),

    # ------------------------------------------------------
    # INSECTICIDE-TREATED NET
    # ------------------------------------------------------

    "insecticidetreated_net_itn__hml10_-1.0",
    "insecticidetreated_net_itn__hml10_0.0",
    "insecticidetreated_net_itn__hml10_1.0",

    # ------------------------------------------------------
    # DHS PLACE OF RESIDENCE
    # ------------------------------------------------------

    "type_of_place_of_residence__hv025_1",
    "type_of_place_of_residence__hv025_2",
]


# ==========================================================
# VALIDATE DEFAULT SCHEMA
# ==========================================================

if len(DEFAULT_FEATURE_COLUMNS) != 66:

    raise ValueError(
        "DEFAULT_FEATURE_COLUMNS must contain exactly "
        f"66 features. Found: {len(DEFAULT_FEATURE_COLUMNS)}"
    )


# ==========================================================
# PROVINCE ENVIRONMENT / GEOGRAPHIC LOOKUP
#
# Used for variables not directly entered by the clinician.
# ==========================================================

PROVINCE_GEO_LOOKUP = {

    "Kigali City": {

        "elevation": 1550,
        "humidity": 77.5,
        "rainfall": 1050,
        "temperature": 19.8,
        "population_density": 6200,
        "health_km": 1.1,
        "access_score": 0.55,
    },

    "Southern Province": {

        "elevation": 1750,
        "humidity": 78.5,
        "rainfall": 1080,
        "temperature": 19.2,
        "population_density": 520,
        "health_km": 2.6,
        "access_score": 0.51,
    },

    "Northern Province": {

        "elevation": 1900,
        "humidity": 79.0,
        "rainfall": 1150,
        "temperature": 17.9,
        "population_density": 480,
        "health_km": 2.8,
        "access_score": 0.50,
    },

    "Eastern Province": {

        "elevation": 1450,
        "humidity": 79.5,
        "rainfall": 950,
        "temperature": 20.3,
        "population_density": 430,
        "health_km": 3.1,
        "access_score": 0.49,
    },

    "Western Province": {

        "elevation": 1650,
        "humidity": 78.0,
        "rainfall": 1400,
        "temperature": 19.0,
        "population_density": 460,
        "health_km": 2.4,
        "access_score": 0.52,
    },
}


# ==========================================================
# FEATURE LABELS
# ==========================================================

FEATURE_LABELS = {

    "has_mosquito_bed_net_for_sleeping__hv227":
        "Mosquito net ownership",

    "has_electricity__hv206":
        "Household electricity",

    "currently_pregnant__sb217":
        "Pregnancy status",

    "highest_educational_level__v106":
        "Education level",

    "wealth_index_combined__hv270":
        "Household socioeconomic proxy score",

    "age_of_household_members__hv105":
        "Age",

    "elevation":
        "Elevation of residence area",

    "annual_humidity":
        "Local annual humidity",

    "annual_rainfall":
        "Local annual rainfall",

    "annual_temperature":
        "Local annual temperature",

    "population_density":
        "Population density",

    "nearest_health_facility_km_fixed":
        "Distance to nearest health facility",

    "health_accessibility_score":
        "Health accessibility score",
}


# ==========================================================
# DHS CATEGORY DESCRIPTIONS
# ==========================================================

WATER_SOURCE_LABELS = {

    "11.0": "Piped water into dwelling",
    "12.0": "Piped water to yard or plot",
    "13.0": "Public tap or standpipe",
    "14.0": "Tube well or borehole",
    "21.0": "Unimproved water source",
    "31.0": "Protected spring",
    "32.0": "Unprotected spring",
    "41.0": "Improved water source",
    "42.0": "Unimproved water source",
    "43.0": "Unimproved water source",
    "51.0": "Rainwater",
    "61.0": "Tanker truck",
    "62.0": "Cart with small tank or drum",
    "71.0": "Surface water",
    "96.0": "Other water source",
}


TOILET_LABELS = {

    "11.0": "Flush toilet connected to sewer system",
    "12.0": "Flush toilet connected to septic system",
    "13.0": "Flush toilet connected elsewhere",
    "14.0": "Flush toilet to pit latrine",
    "21.0": "Ventilated improved pit latrine",
    "22.0": "Improved toilet facility",
    "23.0": "Composting toilet",
    "31.0": "Unimproved toilet facility",
    "41.0": "No toilet facility",
    "96.0": "Other toilet facility",
}


MARITAL_STATUS_LABELS = {

    "0.0": "Never married",
    "1.0": "Married or living with partner",
    "2.0": "Widowed",
    "3.0": "Divorced",
    "4.0": "Separated",
    "5.0": "Other marital status",
}


INSURANCE_LABELS = {

    "0.0": "No health insurance",
    "1.0": "Has health insurance",
    "8.0": "Health insurance status unknown",
}


SPRAY_LABELS = {

    "0.0": "Dwelling not sprayed during the last 12 months",
    "1.0": "Dwelling sprayed during the last 12 months",
    "8.0": "Spraying status unknown",
}


ITN_LABELS = {

    "-1.0": "No mosquito net owned",
    "0.0": "Mosquito net is not insecticide-treated",
    "1.0": "Uses an insecticide-treated mosquito net",
}


EDUCATION_LABELS = {

    0: "No formal education",
    1: "Primary education",
    2: "Secondary education",
    3: "Higher education",
}


# ==========================================================
# BASE VARIABLE GROUP
#
# Groups related one-hot encoded columns into one real-world
# clinical variable.
# ==========================================================

def base_variable_group(col: str) -> str:

    if col.startswith("health_insurance_coverage__"):
        return "health_insurance"

    if col.startswith("insecticidetreated_net_itn__"):
        return "itn_status"

    if col.startswith(
        "has_dwelling_been_sprayed_against_mosquitoes_"
    ):
        return "sprayed_status"

    if col.startswith("current_marital_status__"):
        return "marital_status"

    if col.startswith("source_of_drinking_water__"):
        return "water_source"

    if col.startswith("type_of_toilet_facility__"):
        return "toilet_facility"

    if col.startswith("province_name_"):
        return "province"

    if col.startswith("population_density_category_"):
        return "population_density_category"

    if col.startswith("residence_type_"):
        return "residence_type"

    if col.startswith("type_of_place_of_residence__"):
        return "residence_type"

    return col


# ==========================================================
# CHECK WHETHER FEATURE IS ONE-HOT ENCODED
# ==========================================================

def is_one_hot_feature(col: str) -> bool:

    prefixes = [

        "province_name_",

        "residence_type_",

        "population_density_category_",

        "source_of_drinking_water__",

        "type_of_toilet_facility__",

        "current_marital_status__",

        "health_insurance_coverage__",

        "has_dwelling_been_sprayed_against_mosquitoes_",

        "insecticidetreated_net_itn__",

        "type_of_place_of_residence__",
    ]

    return any(
        col.startswith(prefix)
        for prefix in prefixes
    )


# ==========================================================
# SHOULD SHOW FEATURE IN CLINICIAN EXPLANATION
#
# Inactive one-hot categories should normally not be shown.
#
# Example:
#
# Patient has insurance:
#
#   health_insurance_coverage__sh11_1.0 = 1
#
# The "No insurance" feature is zero and should not be shown
# to the clinician even if SHAP gives that zero-column a value.
# ==========================================================

def should_show_feature(
    col: str,
    value: float,
) -> bool:

    value = float(value)

    if is_one_hot_feature(col):

        return value >= 0.5

    return True


# ==========================================================
# FRIENDLY FEATURE LABEL
# ==========================================================

def friendly_label(col: str) -> str:

    if col in FEATURE_LABELS:

        return FEATURE_LABELS[col]

    if col.startswith("province_name_"):

        province = col.replace(
            "province_name_",
            "",
        )

        return f"Province: {province}"

    if col.startswith(
        "population_density_category_"
    ):

        category = col.replace(
            "population_density_category_",
            "",
        )

        return (
            f"Population density category: {category}"
        )

    if col.startswith("residence_type_"):

        residence = col.replace(
            "residence_type_",
            "",
        )

        return (
            f"Residence type: {residence}"
        )

    if col.startswith(
        "type_of_place_of_residence__hv025_"
    ):

        code = col.rsplit(
            "_",
            1,
        )[-1]

        if code == "1":
            return "Residence type: Urban"

        if code == "2":
            return "Residence type: Rural"

        return "Residence type"

    if col.startswith(
        "source_of_drinking_water__hv201_"
    ):

        code = col.replace(
            "source_of_drinking_water__hv201_",
            "",
        )

        description = WATER_SOURCE_LABELS.get(
            code,
            "Unknown water source",
        )

        return (
            f"Drinking water source: {description}"
        )

    if col.startswith(
        "type_of_toilet_facility__hv205_"
    ):

        code = col.replace(
            "type_of_toilet_facility__hv205_",
            "",
        )

        description = TOILET_LABELS.get(
            code,
            "Unknown toilet facility",
        )

        return (
            f"Toilet facility: {description}"
        )

    if col.startswith(
        "current_marital_status__hv115_"
    ):

        code = col.replace(
            "current_marital_status__hv115_",
            "",
        )

        description = MARITAL_STATUS_LABELS.get(
            code,
            "Unknown marital status",
        )

        return (
            f"Marital status: {description}"
        )

    if col.startswith(
        "health_insurance_coverage__sh11_"
    ):

        code = col.replace(
            "health_insurance_coverage__sh11_",
            "",
        )

        return INSURANCE_LABELS.get(
            code,
            "Health insurance status unknown",
        )

    if col.startswith(
        "has_dwelling_been_sprayed_against_mosquitoes_"
        "in_last_12_months__hv253_"
    ):

        code = col.split(
            "__hv253_"
        )[-1]

        return SPRAY_LABELS.get(
            code,
            "Spraying status unknown",
        )

    if col.startswith(
        "insecticidetreated_net_itn__hml10_"
    ):

        code = col.replace(
            "insecticidetreated_net_itn__hml10_",
            "",
        )

        return ITN_LABELS.get(
            code,
            "ITN status unknown",
        )

    base = (
        col
        .split("__")[0]
        .replace("_", " ")
        .strip()
    )

    return base.capitalize()


# ==========================================================
# CONTEXTUAL CLINICIAN LABEL
#
# Converts encoded feature names into the patient's actual
# real-world description.
# ==========================================================

def contextual_label(
    col: str,
    value: float,
) -> str:

    value = float(value)

    # ------------------------------------------------------
    # AGE
    # ------------------------------------------------------

    if col == "age_of_household_members__hv105":

        return f"Age: {value:g} years"

    # ------------------------------------------------------
    # EDUCATION
    # ------------------------------------------------------

    if col == "highest_educational_level__v106":

        education_value = int(
            round(value)
        )

        education = EDUCATION_LABELS.get(
            education_value,
            f"Education code {education_value}",
        )

        return f"Education: {education}"

        # ------------------------------------------------------
    # SOCIOECONOMIC PROXY
    # ------------------------------------------------------

    if col == "wealth_index_combined__hv270":

        wealth_proxy_labels = {

            1: "Very low socioeconomic proxy level",
            2: "Low socioeconomic proxy level",
            3: "Moderate socioeconomic proxy level",
            4: "High socioeconomic proxy level",
            5: "Very high socioeconomic proxy level",
        }

        proxy_level = int(round(value))

        description = wealth_proxy_labels.get(
            proxy_level,
            "Socioeconomic proxy level unavailable",
        )

        return (
            "Household socioeconomic status: "
            f"{description}"
        )
    # ------------------------------------------------------
    # ENVIRONMENT
    # ------------------------------------------------------

    if col == "elevation":

        return f"Local elevation: {value:g} m"

    if col == "annual_humidity":

        return (
            f"Local annual humidity: {value:g}%"
        )

    if col == "annual_rainfall":

        return (
            f"Local annual rainfall: {value:g} mm"
        )

    if col == "annual_temperature":

        return (
            f"Local annual temperature: {value:g} °C"
        )

    if col == "population_density":

        return (
            "Local population density: "
            f"{value:g}"
        )

    if col == "nearest_health_facility_km_fixed":

        return (
            "Distance to nearest health facility: "
            f"{value:g} km"
        )

    if col == "health_accessibility_score":

        return (
            "Health accessibility score: "
            f"{value:.2f}"
        )

    # ------------------------------------------------------
    # PREGNANCY
    # ------------------------------------------------------

    if col == "currently_pregnant__sb217":

        if value >= 0.5:

            return "Pregnancy status: Pregnant"

        return "Pregnancy status: Not pregnant"

    # ------------------------------------------------------
    # ELECTRICITY
    # ------------------------------------------------------

    if col == "has_electricity__hv206":

        if value >= 0.5:

            return "Household electricity: Yes"

        return "Household electricity: No"

    # ------------------------------------------------------
    # MOSQUITO NET OWNERSHIP
    # ------------------------------------------------------

    if col == (
        "has_mosquito_bed_net_for_sleeping__hv227"
    ):

        if value >= 0.5:

            return "Mosquito net ownership: Yes"

        return "Mosquito net ownership: No"

    # ------------------------------------------------------
    # PROVINCE
    # ------------------------------------------------------

    if col.startswith("province_name_"):

        province = col.replace(
            "province_name_",
            "",
        )

        return f"Province: {province}"

    # ------------------------------------------------------
    # POPULATION DENSITY CATEGORY
    # ------------------------------------------------------

    if col.startswith(
        "population_density_category_"
    ):

        category = col.replace(
            "population_density_category_",
            "",
        )

        return (
            "Population density category: "
            f"{category}"
        )

    # ------------------------------------------------------
    # RESIDENCE TYPE
    # ------------------------------------------------------

    if col.startswith("residence_type_"):

        residence = col.replace(
            "residence_type_",
            "",
        )

        return (
            f"Residence type: {residence}"
        )

    if col.startswith(
        "type_of_place_of_residence__hv025_"
    ):

        code = col.rsplit(
            "_",
            1,
        )[-1]

        if code == "1":

            return "Residence type: Urban"

        if code == "2":

            return "Residence type: Rural"

        return "Residence type: Unknown"

    # ------------------------------------------------------
    # WATER SOURCE
    # ------------------------------------------------------

    if col.startswith(
        "source_of_drinking_water__hv201_"
    ):

        code = col.replace(
            "source_of_drinking_water__hv201_",
            "",
        )

        description = WATER_SOURCE_LABELS.get(
            code,
            "Unknown water source",
        )

        return (
            f"Drinking water source: {description}"
        )

    # ------------------------------------------------------
    # TOILET FACILITY
    # ------------------------------------------------------

    if col.startswith(
        "type_of_toilet_facility__hv205_"
    ):

        code = col.replace(
            "type_of_toilet_facility__hv205_",
            "",
        )

        description = TOILET_LABELS.get(
            code,
            "Unknown toilet facility",
        )

        return (
            f"Toilet facility: {description}"
        )

    # ------------------------------------------------------
    # MARITAL STATUS
    # ------------------------------------------------------

    if col.startswith(
        "current_marital_status__hv115_"
    ):

        code = col.replace(
            "current_marital_status__hv115_",
            "",
        )

        description = MARITAL_STATUS_LABELS.get(
            code,
            "Unknown marital status",
        )

        return (
            f"Marital status: {description}"
        )

    # ------------------------------------------------------
    # HEALTH INSURANCE
    # ------------------------------------------------------

    if col.startswith(
        "health_insurance_coverage__sh11_"
    ):

        code = col.replace(
            "health_insurance_coverage__sh11_",
            "",
        )

        return INSURANCE_LABELS.get(
            code,
            "Health insurance status unknown",
        )

    # ------------------------------------------------------
    # DWELLING SPRAYING
    # ------------------------------------------------------

    if col.startswith(
        "has_dwelling_been_sprayed_against_mosquitoes_"
        "in_last_12_months__hv253_"
    ):

        code = col.split(
            "__hv253_"
        )[-1]

        return SPRAY_LABELS.get(
            code,
            "Spraying status unknown",
        )

    # ------------------------------------------------------
    # ITN STATUS
    # ------------------------------------------------------

    if col.startswith(
        "insecticidetreated_net_itn__hml10_"
    ):

        code = col.replace(
            "insecticidetreated_net_itn__hml10_",
            "",
        )

        return ITN_LABELS.get(
            code,
            "ITN status unknown",
        )

    # ------------------------------------------------------
    # GENERAL FALLBACK
    # ------------------------------------------------------

    return friendly_label(col)


# ==========================================================
# LOAD FEATURE COLUMNS
# ==========================================================

def _load_feature_columns():

    global _feature_columns

    if os.path.exists(COLUMNS_PATH):

        with open(
            COLUMNS_PATH,
            "r",
            encoding="utf-8",
        ) as f:

            columns = json.load(f)

        print(
            "Loaded feature schema from JSON:"
        )

        print(
            f"Number of features: {len(columns)}"
        )

    else:

        columns = (
            DEFAULT_FEATURE_COLUMNS.copy()
        )

        print(
            "feature_columns.json not found."
        )

        print(
            "Using built-in 66-feature schema."
        )

    if len(columns) != 66:

        raise ValueError(
            "MARISA feature schema must contain exactly "
            f"66 features. Found: {len(columns)}"
        )

    _feature_columns = list(columns)

    return _feature_columns


# ==========================================================
# LOAD MODEL
# ==========================================================

def load_model():

    global _model
    global _feature_columns
    global _is_placeholder

    # ------------------------------------------------------
    # RETURN ALREADY LOADED MODEL
    # ------------------------------------------------------

    if _model is not None:

        return _model

    # ------------------------------------------------------
    # CHECK MODEL FILE
    # ------------------------------------------------------

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            "\n\n"
            "==================================================\n"
            "MARISA PRODUCTION MODEL NOT FOUND\n"
            "==================================================\n"
            f"Expected model path:\n{MODEL_PATH}\n\n"
            "Expected model filename:\n"
            "rf_final_model.pkl\n\n"
            "Copy the trained model into:\n"
            "backend/app/artifacts/\n"
            "==================================================\n"
        )

    print()
    print("=" * 60)
    print("LOADING MARISA FINAL RANDOM FOREST MODEL")
    print("=" * 60)

    print()
    print("Model path:")
    print(MODEL_PATH)

    # ------------------------------------------------------
    # LOAD MODEL
    # ------------------------------------------------------

    _model = joblib.load(MODEL_PATH)

    _is_placeholder = False

    print()
    print("Model loaded successfully.")

    print()
    print("MODEL TYPE:")
    print(type(_model))

    print()
    print("MODEL DETAILS:")
    print(_model)

    # ------------------------------------------------------
    # VALIDATE MODEL TYPE
    # ------------------------------------------------------

    if not isinstance(
        _model,
        RandomForestClassifier,
    ):

        raise TypeError(
            "\n"
            "Incorrect model type loaded.\n"
            "Expected:\n"
            "RandomForestClassifier\n\n"
            "Received:\n"
            f"{type(_model)}\n"
        )

    # ------------------------------------------------------
    # VALIDATE FEATURE COUNT
    # ------------------------------------------------------

    if not hasattr(
        _model,
        "n_features_in_",
    ):

        raise AttributeError(
            "Loaded model does not contain "
            "n_features_in_."
        )

    expected_features = int(
        _model.n_features_in_
    )

    print()
    print("MODEL EXPECTED FEATURES:")
    print(expected_features)

    if expected_features != 66:

        raise ValueError(
            "Unexpected MARISA model feature count.\n"
            "Expected 66 features.\n"
            f"Model expects {expected_features}."
        )

    # ------------------------------------------------------
    # LOAD FEATURE SCHEMA
    # ------------------------------------------------------

    file_columns = _load_feature_columns()

    # ------------------------------------------------------
    # MODEL FEATURE NAMES ARE SOURCE OF TRUTH
    # ------------------------------------------------------

    if hasattr(
        _model,
        "feature_names_in_",
    ):

        model_columns = (
            _model.feature_names_in_.tolist()
        )

        print()
        print(
            "Feature names found inside model."
        )

        if file_columns != model_columns:

            print()
            print("WARNING:")

            print(
                "feature_columns.json / default schema "
                "does not exactly match the feature order "
                "stored inside rf_final_model.pkl."
            )

            print()

            print(
                "The model's embedded feature order "
                "will be used as the source of truth."
            )

        _feature_columns = model_columns

    else:

        _feature_columns = file_columns

    # ------------------------------------------------------
    # FINAL VALIDATION
    # ------------------------------------------------------

    if len(_feature_columns) != expected_features:

        raise ValueError(
            "\n"
            "FINAL FEATURE SCHEMA MISMATCH\n"
            f"Model expects: {expected_features}\n"
            "Loaded schema contains: "
            f"{len(_feature_columns)}\n"
        )

    print()
    print(
        "FINAL FEATURE SCHEMA VALIDATED:"
    )

    print(
        f"{len(_feature_columns)} features"
    )

    print()
    print("MARISA MODEL READY")
    print("=" * 60)
    print()

    return _model


# ==========================================================
# PLACEHOLDER STATUS
# ==========================================================

def is_placeholder():

    return _is_placeholder


# ==========================================================
# NORMALIZE INPUT
# ==========================================================

def _normalize_string(value):

    if value is None:

        return ""

    return str(value).strip().lower()


# ==========================================================
# ENCODE PATIENT
# ==========================================================

def encode_patient(patient) -> pd.DataFrame:

    global _model
    global _feature_columns

    # ------------------------------------------------------
    # ENSURE MODEL IS LOADED
    # ------------------------------------------------------

    if _model is None:

        load_model()

    columns = _feature_columns

    if columns is None:

        raise RuntimeError(
            "Feature columns have not been loaded."
        )

    # ------------------------------------------------------
    # START WITH ZERO VECTOR
    # ------------------------------------------------------

    row = pd.Series(
        0.0,
        index=columns,
        dtype=float,
    )

    # ======================================================
    # PROVINCE
    # ======================================================

    province = str(
        patient.province
    ).strip()

    if province not in PROVINCE_GEO_LOOKUP:

        raise ValueError(
            f"Unknown province: {province}. "
            "Expected one of: "
            f"{list(PROVINCE_GEO_LOOKUP.keys())}"
        )

    geo = PROVINCE_GEO_LOOKUP[province]

    # ======================================================
    # GEOGRAPHIC FEATURES
    # ======================================================

    row["elevation"] = geo["elevation"]

    row["annual_humidity"] = geo["humidity"]

    row["annual_rainfall"] = geo["rainfall"]

    row["annual_temperature"] = (
        geo["temperature"]
    )

    row["population_density"] = (
        geo["population_density"]
    )

    row[
        "nearest_health_facility_km_fixed"
    ] = geo["health_km"]

    row[
        "health_accessibility_score"
    ] = geo["access_score"]

    # ======================================================
    # AGE
    # ======================================================

    row[
        "age_of_household_members__hv105"
    ] = float(patient.age)

    # ======================================================
    # PREGNANCY
    # ======================================================

    row[
        "currently_pregnant__sb217"
    ] = (
        1.0
        if patient.pregnant
        else 0.0
    )

    # ======================================================
    # ELECTRICITY
    # ======================================================

    row[
        "has_electricity__hv206"
    ] = (
        1.0
        if patient.electricity
        else 0.0
    )

    # ======================================================
    # MOSQUITO NET OWNERSHIP
    # ======================================================

    row[
        "has_mosquito_bed_net_for_sleeping__hv227"
    ] = (
        1.0
        if patient.has_net
        else 0.0
    )

    # ======================================================
    # EDUCATION
    # ======================================================

    education = _normalize_string(
        patient.education
    )

    education_map = {

        "none": 0,
        "primary": 1,
        "secondary": 2,
        "higher": 3,
    }

    row[
        "highest_educational_level__v106"
    ] = float(
        education_map.get(
            education,
            1,
        )
    )

    # ======================================================
    # NORMALIZE SOCIOECONOMIC INPUTS
    # ======================================================

    water_source = _normalize_string(
        patient.water_source
    )

    toilet_facility = _normalize_string(
        patient.toilet_facility
    )

    health_insurance = _normalize_string(
        patient.health_insurance
    )

    # ======================================================
    # SOCIOECONOMIC PROXY
    #
    # This is NOT the original DHS wealth index.
    #
    # It is a simple proxy constructed from clinician-facing
    # variables available to this API.
    # ======================================================

    wealth_points = sum([

        bool(patient.electricity),

        water_source == "improved",

        toilet_facility == "improved",

        health_insurance == "yes",
    ])

    row[
        "wealth_index_combined__hv270"
    ] = float(
        1 + wealth_points
    )

    # ======================================================
    # PROVINCE ONE-HOT
    # ======================================================

    province_column = (
        f"province_name_{province}"
    )

    if province_column in row.index:

        row[
            province_column
        ] = 1.0

    # ======================================================
    # RESIDENCE TYPE
    # ======================================================

    residence_type = str(
        patient.residence_type
    ).strip()

    if residence_type not in [
        "Rural",
        "Urban",
    ]:

        raise ValueError(
            "residence_type must be "
            "'Rural' or 'Urban'."
        )

    residence_column = (
        f"residence_type_{residence_type}"
    )

    if residence_column in row.index:

        row[
            residence_column
        ] = 1.0

    # ======================================================
    # DHS PLACE OF RESIDENCE
    #
    # 1 = Urban
    # 2 = Rural
    # ======================================================

    place_code = (
        "1"
        if residence_type == "Urban"
        else "2"
    )

    place_column = (
        "type_of_place_of_residence__hv025_"
        f"{place_code}"
    )

    if place_column in row.index:

        row[
            place_column
        ] = 1.0

    # ======================================================
    # POPULATION DENSITY CATEGORY
    # ======================================================

    if province == "Kigali City":

        density_category = (
            "Very High Density"
        )

    elif province in [

        "Southern Province",
        "Northern Province",
    ]:

        density_category = (
            "High Density"
        )

    else:

        density_category = (
            "Medium Density"
        )

    density_column = (
        "population_density_category_"
        f"{density_category}"
    )

    if density_column in row.index:

        row[
            density_column
        ] = 1.0

    # ======================================================
    # DRINKING WATER
    # ======================================================

    if water_source == "improved":

        water_code = "41.0"

    else:

        water_code = "21.0"

    water_column = (
        "source_of_drinking_water__hv201_"
        f"{water_code}"
    )

    if water_column in row.index:

        row[
            water_column
        ] = 1.0

    # ======================================================
    # TOILET FACILITY
    # ======================================================

    if toilet_facility == "improved":

        toilet_code = "22.0"

    else:

        toilet_code = "31.0"

    toilet_column = (
        "type_of_toilet_facility__hv205_"
        f"{toilet_code}"
    )

    if toilet_column in row.index:

        row[
            toilet_column
        ] = 1.0

    # ======================================================
    # MARITAL STATUS
    # ======================================================

    marital_status = _normalize_string(
        patient.marital_status
    )

    marital_map = {
        "never": "0.0",
        "married": "1.0",
        "widowed": "2.0",
        "divorced": "3.0",
        "separated": "4.0",
    }

    marital_code = marital_map.get(
        marital_status,
        "1.0",
    )

    marital_column = (
        "current_marital_status__hv115_"
        f"{marital_code}"
    )

    if marital_column in row.index:

        row[
            marital_column
        ] = 1.0

    # ======================================================
    # HEALTH INSURANCE
    # ======================================================

    insurance_map = {

        "yes": "1.0",
        "no": "0.0",
        "unknown": "8.0",
    }

    insurance_code = insurance_map.get(
        health_insurance,
        "8.0",
    )

    insurance_column = (
        "health_insurance_coverage__sh11_"
        f"{insurance_code}"
    )

    if insurance_column in row.index:

        row[
            insurance_column
        ] = 1.0

    # ======================================================
    # HOUSEHOLD SPRAYING
    # ======================================================

    sprayed_last_12mo = _normalize_string(
        patient.sprayed_last_12mo
    )

    spray_map = {

        "yes": "1.0",
        "no": "0.0",
        "unknown": "8.0",
    }

    spray_code = spray_map.get(
        sprayed_last_12mo,
        "8.0",
    )

    spray_column = (
        "has_dwelling_been_sprayed_against_"
        "mosquitoes_in_last_12_months__hv253_"
        f"{spray_code}"
    )

    if spray_column in row.index:

        row[
            spray_column
        ] = 1.0

    # ======================================================
    # INSECTICIDE-TREATED NET
    # ======================================================

    if not patient.has_net:

        itn_code = "-1.0"

    elif patient.itn_treated is True:

        itn_code = "1.0"

    else:

        itn_code = "0.0"

    itn_column = (
        "insecticidetreated_net_itn__hml10_"
        f"{itn_code}"
    )

    if itn_column in row.index:

        row[
            itn_column
        ] = 1.0

    # ======================================================
    # CREATE FINAL DATAFRAME
    # ======================================================

    X = pd.DataFrame(
        [row],
        columns=columns,
    )

    # ======================================================
    # FORCE NUMERIC DATA
    # ======================================================

    X = X.apply(
        pd.to_numeric,
        errors="coerce",
    ).fillna(0.0)

    # ======================================================
    # FEATURE COUNT VALIDATION
    # ======================================================

    if X.shape[1] != 66:

        raise ValueError(
            "Encoded patient feature count is incorrect.\n"
            "Expected: 66\n"
            f"Received: {X.shape[1]}"
        )

    # ======================================================
    # EXACT MODEL FEATURE ORDER VALIDATION
    # ======================================================

    if hasattr(
        _model,
        "feature_names_in_",
    ):

        expected_columns = (
            _model.feature_names_in_.tolist()
        )

        if (
            X.columns.tolist()
            != expected_columns
        ):

            raise ValueError(
                "\n"
                "CRITICAL FEATURE ORDER MISMATCH\n"
                "The encoded patient DataFrame does not "
                "match the feature order expected by "
                "rf_final_model.pkl.\n"
            )

    return X


# ==========================================================
# RISK CLASSIFICATION
# ==========================================================

def classify_risk(
    score: float,
) -> str:
    """
    Convert the Random Forest model score into a clinician-
    friendly risk category.

    These categories are used for risk communication and
    clinical decision support.

    They do not represent a malaria diagnosis.
    """

    score = float(score)

    if score < 0.10:

        return "Low"

    if score < 0.20:

        return "Moderate"

    if score < 0.35:

        return "High"

    return "Very High"