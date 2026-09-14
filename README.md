# MARISA — Explainable AI Clinical Decision Support System

Production-oriented implementation of the MARISA research: an explainable
Random Forest model for malaria risk prediction among women of reproductive
age in Rwanda, served through a real API and two purpose-built frontends.

## Architecture

```
                     ┌─────────────────────┐
                     │   PostgreSQL DB      │  ← every prediction logged
                     │  (prediction_logs)   │     for audit/drift/fairness
                     └──────────▲───────────┘
                                │
                     ┌──────────┴───────────┐
                     │   FastAPI Backend     │  /predict  /health  /drift-summary
                     │  (Random Forest + SHAP)│
                     └────┬─────────────┬────┘
                          │             │
              ┌───────────▼───┐   ┌─────▼──────────┐
              │   Streamlit    │   │   Dash (Plotly) │
              │  Clinician Tool │   │  Admin Dashboard │
              │  (risk score +  │   │ (multi-clinic    │
              │  SHAP explain)  │   │  monitoring)      │
              └────────────────┘   └──────────────────┘
```

## Why this structure (not just one app)

- **FastAPI backend** is the single source of truth for predictions — both frontends call the *same* model and get *identical* results, and the model can be updated in one place.
- **PostgreSQL logging** turns every prediction into an auditable record — this is what makes `/drift-summary` possible, and what a real deployment needs for fairness review (e.g., "is the model flagging women in one province disproportionately more than another?").
- **Streamlit vs. Dash split** reflects two genuinely different users: a nurse assessing one patient at a time (Streamlit, simple form-in/result-out) vs. a supervisor or MOH analyst watching aggregate patterns across clinics (Dash, built for live-refreshing analytics dashboards).

## Running locally (no Docker — fastest way to test each piece)

```bash
# 1. Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
# Visit http://localhost:8000/docs for interactive API docs

# 2. Streamlit (in a new terminal)
cd streamlit_app
pip install streamlit plotly requests
API_BASE_URL=http://localhost:8000 streamlit run app.py

# 3. Dash (in a new terminal)
cd dash_app
pip install dash plotly requests
API_BASE_URL=http://localhost:8000 python app.py
```

## Running everything together with Docker Compose

```bash
docker compose up --build
```

- Backend API + docs: http://localhost:8000/docs
- Streamlit clinician tool: http://localhost:8501
- Dash admin dashboard: http://localhost:8050
- PostgreSQL: localhost:5432 (credentials configured through environment variables)


## Plugging in your REAL trained model (important — currently runs a placeholder)

The backend currently runs a small **placeholder model** trained on synthetic
data so the whole system is runnable and demoable immediately. To serve your
actual research model:

1. In your Objective 3/4 notebook (where `rf_calibrated`, `X_train` exist), run the script `export_model_artifacts.py` from this repo — it saves `rf_calibrated.pkl` and `feature_columns.json`.
2. Copy both files into `backend/app/artifacts/`.
3. Restart the backend (`uvicorn ... --reload` will pick it up automatically; with Docker, run `docker compose up --build backend`).
4. Check `/health` — `model_version` should no longer show `[PLACEHOLDER]`.

**Important:** use `rf_calibrated` (the isotonic-calibrated model), not the raw `rf_final` — the calibration analysis in this project found the uncalibrated model was substantially overconfident at higher risk levels.

## Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/predict` | POST | Risk score + SHAP-based top factors for one patient |
| `/health` | GET | Service and model status check |
| `/drift-summary` | GET | Aggregate stats across all logged predictions — mean risk, high-risk rate, breakdown by province, overall vs. last 7 days |

## Known limitations (state these explicitly in your defense — this is a strength, not a weakness)

- The feature encoder (`ml_model.encode_patient`) fills geographic/climate variables (elevation, rainfall, humidity, etc.) using **province-level averages**, since a nurse cannot measure these at point of care. This is a documented simplification, not a hidden approximation.
- Water source and toilet facility inputs are simplified to improved/unimproved and mapped to one representative DHS sub-category each, rather than the full set of DHS categories used in training.
- Per the project's Objective 5 findings, **the model's validation specifically for pregnant women is limited** (23 confirmed cases, unstable cross-validated recall ~11%) — the API surfaces this caveat automatically on every prediction where `pregnant=true`.

## Not yet built (documented, future work)

- **PyInstaller offline `.exe`** — for clinics with no internet at all, wrapping the Streamlit app + a local SQLite fallback (already supported by `database.py`) into a single executable. Not attempted yet; the Streamlit + SQLite combination is the right foundation for this when you're ready.
- **Authentication / multi-clinic accounts** — currently `clinic_id` is a free-text field for basic segmentation, not a real login system.
- **HTTPS / production secrets management — production deployments must use HTTPS and securely managed environment variables or a secrets manager. Database credentials must never be committed to the repository.

## Presentation materials

- `python-pptx` — for the lightning talk and defense slide decks (separate from this codebase; ask if you want a script to auto-generate slides summarizing the model comparison table and SHAP findings)
- LaTeX (Beamer + TikZ) — for the formal academic presentation and framework diagram (the architecture diagram above can be redrawn in TikZ for that version)
