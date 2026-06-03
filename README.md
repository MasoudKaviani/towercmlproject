# 🗼 TowerCML — Telecom Tower Fault Detection

Continuous Machine Learning pipeline for predicting telecom tower faults (`tower_status`).
Built with **Python · scikit-learn · DVC · GitHub Actions CML · FastAPI · Docker**.

---

## Architecture

```
data/raw/  ──► preprocess ──► data/processed/  ──► train ──► models/  ──► evaluate ──► reports/
                                                                                          │
                                                                                    GitHub Actions
                                                                                    CML PR comment
                                                                                    (metrics + plots)
                                                                                          │
                                                                                    FastAPI /predict
```

## Project Structure

```
TowerCML/
├── .github/workflows/cml.yml   # GitHub Actions: train → evaluate → CML report → Docker push
├── api/
│   └── main.py                 # FastAPI REST API
├── src/
│   ├── preprocess.py           # DVC stage 1: clean & encode features
│   ├── train.py                # DVC stage 2: train Random Forest
│   └── evaluate.py             # DVC stage 3: metrics + plots
├── data/
│   └── raw/                    # DVC-tracked raw data
├── models/                     # DVC-tracked model artifacts
├── reports/                    # metrics.json + PNG plots
├── params.yaml                 # All hyperparameters (DVC params)
├── dvc.yaml                    # DVC pipeline definition
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/<your-org>/TowerCML.git
cd TowerCML
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Initialize DVC

```bash
dvc init
git add .dvc .dvcignore
git commit -m "chore: init DVC"
```

### 3. Track the dataset with DVC

```bash
dvc add data/raw/telecom_tower_dataset.csv
git add data/raw/telecom_tower_dataset.csv.dvc .gitignore
git commit -m "data: track raw dataset with DVC"
```

*(Optional) Add a remote storage (S3, GCS, Azure, etc.) so CI can pull the data:*

```bash
dvc remote add -d myremote s3://your-bucket/tower-dvc
dvc push
```

### 4. Run the full pipeline locally

```bash
dvc repro
```

This runs all three stages: `preprocess → train → evaluate`.
Outputs land in `data/processed/`, `models/`, and `reports/`.

### 5. Check results

```bash
dvc metrics show          # print metrics.json
dvc metrics diff          # compare to last commit
dvc plots show            # open plots in browser
```

---

## FastAPI Server

### Run locally

```bash
uvicorn api.main:app --reload
```

Open the interactive docs: **http://localhost:8000/docs**

### Key endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/metrics` | Latest evaluation metrics |
| POST | `/predict` | Single tower prediction |
| POST | `/predict/batch` | Batch predictions (≤500) |

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tower_temperature": 72.3,
    "input_voltage": 185.0,
    "current_consumption": 85.0,
    "wind_speed": 55.0,
    "humidity": 90.0,
    "connected_users": 500,
    "data_traffic_gb": 150.0,
    "signal_strength": -95.0,
    "power_outages_24h": 8,
    "tower_age_years": 12,
    "active_antennas": 4,
    "backup_battery_charge": 15.0,
    "packet_error_rate": 18.5,
    "days_since_maintenance": 400,
    "tower_type": "3G"
  }'
```

### Run with Docker

```bash
docker compose up --build
```

---

## GitHub Actions / CML Setup

The workflow in `.github/workflows/cml.yml` runs automatically on every push to `main` or on any PR.

### Required GitHub Secrets

| Secret | Purpose |
|--------|---------|
| `GITHUB_TOKEN` | Auto-provided by GitHub — used by CML to post PR comments |
| `AWS_ACCESS_KEY_ID` *(optional)* | DVC remote on S3 |
| `AWS_SECRET_ACCESS_KEY` *(optional)* | DVC remote on S3 |

### What the workflow does

1. **Install** Python 3.11 + dependencies + CML CLI
2. **`dvc repro`** — runs the full pipeline (preprocess → train → evaluate)
3. **Post a CML comment** on the PR with:
   - Metrics table (accuracy, F1, precision, recall, AUC)
   - ROC curve plot
   - Confusion matrix plot
   - Feature importance plot
   - Param & metric diffs vs previous commit
4. **Upload** model artifacts (`.pkl` files, metrics, plots) as GitHub Actions artifacts
5. **Build & push** Docker image to `ghcr.io` (main branch only)

---

## Tuning Hyperparameters

Edit `params.yaml` and re-run `dvc repro`:

```yaml
train:
  n_estimators: 300     # more trees
  max_depth: 20
  class_weight: balanced  # handles 10:1 class imbalance
```

DVC tracks which params changed and only re-runs affected stages.

---

## Model Notes

- **Target**: `tower_status` — binary (0 = Normal, 1 = Fault)
- **Class imbalance**: ~10:1 (4548 normal vs 452 fault) — handled via `class_weight='balanced'`
- **Features**: 14 numeric + 1 one-hot encoded categorical (`tower_type`)
- **Evaluation**: F1, Precision, Recall, ROC-AUC (accuracy alone misleads on imbalanced data)
