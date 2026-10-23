# ML Model Deployment Strategies and Model Serving using FastAPI

A Titanic survival classification assignment demonstrating online, batch, streaming, and REST API inference.

## Dataset

Download `train.csv` from https://www.kaggle.com/competitions/titanic/data and place it at `data/train.csv`.

## Setup

Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Train once

```bash
python -m src.train_model
```

The script performs a stratified 80:20 split, fits preprocessing and a Random Forest classifier on the training portion, evaluates accuracy on the held-out portion, saves the complete pipeline to `models/model.pkl`, and writes feature-only held-out records to `data/unseen_passengers.csv`.

## Run inference strategies

```bash
python -m src.online_inference
python -m src.batch_inference
python -m src.streaming_inference
```

Batch output is saved to `data/batch_predictions.csv`.

## Start FastAPI

```bash
uvicorn src.app:app --reload
```

Open http://127.0.0.1:8000/docs to test `/health` and `/predict`. A valid request body:

```json
{
  "Pclass": 3,
  "Sex": "male",
  "Age": 22,
  "SibSp": 1,
  "Parch": 0,
  "Fare": 7.25,
  "Embarked": "S"
}
```

For an invalid request, try `Pclass: 5` or `Sex: "unknown"`; Pydantic should return HTTP 422. With the API still running, open another terminal and run:

```bash
python -m src.client
```

## Observations

1. **Online vs batch:** Online predicts one record per request; batch predicts a group of records together.
2. **Streaming vs batch:** Streaming processes records as they arrive, one at a time; batch groups records and processes them together.
3. **Why FastAPI:** It exposes predictions through REST endpoints, validates input using Pydantic, and provides interactive Swagger documentation.
4. **Why reuse one model:** The same fitted pipeline keeps preprocessing and predictions consistent across all inference approaches and avoids model differences caused by retraining.

## Submission checklist

- [ ] `models/model.pkl`
- [ ] `data/batch_predictions.csv`
- [ ] Screenshots of online, batch, and streaming outputs
- [ ] Screenshots of `/health`, valid `/predict`, and invalid `/predict`
- [ ] `report/assignment_report.pdf`
- [ ] Public GitHub repository with all required files

Do not commit secrets, tokens, or passwords.
