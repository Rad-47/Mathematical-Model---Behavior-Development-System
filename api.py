
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from scorer import score_session

app = FastAPI(title="BCAT–Spiky Cultural Scoring API", version="1.0.0")

class SpikyIn(BaseModel):
    language: Optional[Dict[str, Any]] = None
    vocal: Optional[Dict[str, Any]] = None
    facial: Optional[Dict[str, Any]] = None
    interaction: Optional[Dict[str, Any]] = None
    highlevel: Optional[Dict[str, Any]] = None

class ScoreRequest(BaseModel):
    spiky: SpikyIn = Field(..., description="Spiky metrics payload")
    bcat_pattern: Optional[List[str]] = Field(None, description="Ordered high→low, e.g. ['Resolve','Precision','Harmony','Innovation']")
    pattern_id: Optional[int] = Field(None, description="BCAT 24-pattern ID, 1..24")
    pattern_name: Optional[str] = Field(None, description="Optional exact pattern name from config")

class ScoreResponse(BaseModel):
    factors: Dict[str, float]
    alignment_pct: float
    top_drivers: List[Dict[str, Any]]
    normalized_metrics: Dict[str, float]

@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    pattern_selector = req.pattern_id if req.pattern_id is not None else (req.pattern_name if req.pattern_name else req.bcat_pattern)
    result = score_session(req.spiky.dict(exclude_none=True), pattern_selector)
    return result

@app.get("/health")
def health():
    return {"ok": True}
