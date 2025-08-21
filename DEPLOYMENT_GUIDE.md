# Deployment Guide — BCAT Alignment Service

## Local (dev)
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn api:app --reload --port 8080
```

## Docker
```bash
docker build -t bcat-align:latest .
docker run --rm -p 8080:8080 bcat-align:latest
```

## Cloud
- Cloud Run / Fly.io / Render / Heroku: deploy the container and expose 8080.
- AWS ECS/Fargate: push image to ECR, run a service with ALB on 8080.

## Health
- `GET /health` → `{ "ok": true }`

## Config-only tuning
- `config/weights.json` — metric→factor weights (rows sum to 1)
- `config/multipliers.json` — primary/secondary/tertiary/quaternary strengths
- `config/patterns.json` — 24 patterns (IDs + order)
