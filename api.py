from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, List
from scorer import score_payload, load_patterns_by

app = FastAPI(title="BCAT Alignment Service", version="1.0.0")

class ScoreRequest(BaseModel):
    spiky: Dict
    pattern_id: Optional[int] = None
    pattern_name: Optional[str] = None
    bcat_pattern: Optional[List[str]] = None
    session_id: Optional[str] = None
    team_id: Optional[str] = None
    user_id: Optional[str] = None
    role: Optional[str] = None

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/score")
def score(req: ScoreRequest):
    pattern = load_patterns_by(req.pattern_id, req.pattern_name, req.bcat_pattern)
    res = score_payload(req.spiky, pattern)
    return JSONResponse(res)
