"""
Run this in your Objective 3/4 notebook (where rf_calibrated, X_train,
and X_test already exist in memory) to export the two files the API
needs to serve YOUR real trained model instead of the placeholder.

After running this, copy the two output files into:
    backend/app/artifacts/rf_calibrated.pkl
    backend/app/artifacts/feature_columns.json

Then restart the backend (or docker compose up --build) and it will
automatically detect and load your real model - no code changes needed.
"""
import json
import joblib

# ==========================================================
# 1. Save the calibrated model
#    (use rf_calibrated from the calibration step - NOT rf_final,
#    given the calibration analysis showed the raw model is
#    systematically overconfident)
# ==========================================================
joblib.dump(rf_calibrated, "rf_calibrated.pkl")
print("Saved rf_calibrated.pkl")

# ==========================================================
# 2. Save the exact feature column order used during training
#    This MUST match X_train.columns exactly, in the same order,
#    or predictions will be silently wrong (misaligned columns)
# ==========================================================
with open("feature_columns.json", "w") as f:
    json.dump(X_train.columns.tolist(), f, indent=2)
print("Saved feature_columns.json with", len(X_train.columns), "columns")

print("\nNow copy both files into: backend/app/artifacts/")
print("Then restart the backend - it will load your real model automatically.")
