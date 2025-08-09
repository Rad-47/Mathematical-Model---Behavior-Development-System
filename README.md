
# BCAT–Spiky Cultural Scoring (Reference Model v1)

A small, config-driven service that converts **Spiky.ai** conversation metrics into **BCAT**-aligned
behavior factor scores (Precision, Resolve, Innovation, Harmony) and an **Alignment %** against a team’s
North Star pattern.

## Features
- Configurable metric→factor weights (`config/weights.json`)
- Pattern multipliers (`config/multipliers.json`)
- Normalization for common metrics (ratios, WPM, patience, etc.)
- Explainability: per-metric contributions ("top_drivers")
- REST API via FastAPI (`/score`)
- No external data calls; stateless

## Install
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
uvicorn api:app --host 0.0.0.0 --port 8080 --reload
```
Health check: `GET /health`

## Score Example
```bash
curl -X POST http://localhost:8080/score \
  -H "Content-Type: application/json" \
  -d @examples/example_request.json
```

## Files
- `scorer.py` – scoring logic (normalization, weights, multipliers, alignment)
- `api.py` – FastAPI app exposing `/score`
- `config/weights.json` – metric→factor weights (rows sum to 1)
- `config/multipliers.json` – BCAT pattern multipliers
- `examples/` – example request/response

## Customize
- Update `config/weights.json` according to BCAT expert inputs.
- Change multipliers in `config/multipliers.json` if your pattern priorities differ.
- Extend normalizers in `scorer.normalize_metrics` for new metrics.

## Integration
- Upstream system (FreeFuse) sends Spiky JSON + BCAT pattern.
- Receive factor scores + alignment %; store for dashboards.

## License
Reference implementation for pilot use.
Generated 2025-08-09.


## Using Pattern IDs
You can now pass a **pattern_id (1..24)** instead of the explicit factor order.
Example:

```bash
curl -X POST http://localhost:8080/score \
  -H "Content-Type: application/json" \
  -d '{
  "spiky": { "language": {"positivity":72, "objectivity":64, "filler_ratio":0.1, "hedging_ratio":0.15}},
  "pattern_id": 7
}'
```

Where **7 = "Resolve → Precision → Innovation → Harmony"** (see `config/patterns.json`).

You may also pass `pattern_name` with the exact string in `patterns.json`.
