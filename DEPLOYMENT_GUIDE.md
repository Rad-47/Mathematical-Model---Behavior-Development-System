
# BCAT–Spiky Cultural Scoring — One‑Page Deployment Guide

## What this service does
Converts **Spiky.ai** conversation metrics into **BCAT**-aligned factor scores (Precision, Resolve, Innovation, Harmony) and an **Alignment %** versus the target BCAT pattern. Outputs include **top drivers** for coaching.

---

## Run Locally (no Docker)
1) **Python 3.11+**
2) Install deps:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3) Start API:
   ```bash
   uvicorn api:app --host 0.0.0.0 --port 8080
   ```
4) Health check: `GET http://localhost:8080/health`

## Run with Docker (recommended for handoff)
```bash
docker build -t bcat-spiky .
docker run -p 8080:8080 --name bcat-spiky bcat-spiky
```
- API available at `http://localhost:8080`

---

## Configuration
- **Metric → Factor weights**: `config/weights.json` (rows sum to 1 across factors)
- **Pattern multipliers**: `config/multipliers.json` (primary/secondary/tertiary/quaternary)
- **24 BCAT patterns**: `config/patterns.json` (IDs 1..24)

Update configs without code changes. Restart the service to apply.

---

## API Usage

### Score by Pattern ID (easiest)
```bash
curl -X POST http://localhost:8080/score   -H "Content-Type: application/json"   -d '{
    "spiky": {
      "language": { "positivity": 72, "objectivity": 64, "filler_ratio": 0.10, "hedging_ratio": 0.15 }
    },
    "pattern_id": 7
  }'
```
*Pattern 7 = "Resolve → Precision → Innovation → Harmony".*

### Score by Explicit Pattern Order
```bash
curl -X POST http://localhost:8080/score   -H "Content-Type: application/json"   -d '{
    "spiky": {
      "language": { "positivity": 72, "objectivity": 64, "filler_ratio": 0.10, "hedging_ratio": 0.15 }
    },
    "bcat_pattern": ["Resolve","Precision","Innovation","Harmony"]
  }'
```

### Response (fields)
- `factors`: `{ precision, resolve, innovation, harmony }` (0–100)
- `alignment_pct`: overall fit to target pattern (0–100)
- `top_drivers`: contributions by metric → factor (explainability)
- `normalized_metrics`: inputs after normalization

---

## Data Expectations
Send Spiky metrics in the structure shown in `examples/example_request.json`. Normalization rules (ratios, inversions, caps) are implemented in `scorer.py::normalize_metrics`.

---

## Security & Ops
- No external calls; stateless. Avoid sending PII—use IDs.
- Put behind your API gateway; add auth if exposed beyond VPC.
- Logging: add your preferred middleware if needed.
- Versioning: track config changes (`weights.json`) in git.

---

## Support
- Adjust mapping with BCAT experts (`config/weights.json`).
- Align Spiky payload fields to `examples/example_request.json`.
- Questions: owners of the FreeFuse integration.
