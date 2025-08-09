
from typing import Dict, List, Tuple
import math
import json
import pathlib

CONFIG_DIR = pathlib.Path(__file__).parent / "config"

def load_config():
    with open(CONFIG_DIR / "weights.json") as f:
        weights = json.load(f)
    with open(CONFIG_DIR / "multipliers.json") as f:
        multipliers = json.load(f)
    # Build lookup
    weight_lookup = {row["metric"]: {k: row[k] for k in ["precision","resolve","innovation","harmony"]} for row in weights}
    return weight_lookup, multipliers

WEIGHTS, MULTS = load_config()

import itertools

def load_patterns():
    p = json.load(open(CONFIG_DIR / "patterns.json"))
    id_to_order = {int(row["id"]): row["order"] for row in p}
    name_to_order = {row["name"]: row["order"] for row in p}
    return id_to_order, name_to_order

PATTERN_BY_ID, PATTERN_BY_NAME = load_patterns()

def resolve_pattern(pattern_or_id):
    """
    Accepts:
      - list of factors, e.g., ["Resolve","Precision","Harmony","Innovation"]
      - integer id (1..24)
      - pattern name string (exact match from config)
    Returns normalized order list.
    """
    if isinstance(pattern_or_id, list):
        return pattern_or_id
    if isinstance(pattern_or_id, int):
        return PATTERN_BY_ID.get(pattern_or_id)
    if isinstance(pattern_or_id, str):
        if pattern_or_id in PATTERN_BY_NAME:
            return PATTERN_BY_NAME[pattern_or_id]
        # also allow comma-separated "Resolve, Precision, Harmony, Innovation"
        parts = [p.strip() for p in pattern_or_id.split(",")]
        if len(parts) == 4:
            return parts
    return None

# ---- Normalization helpers ----

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def scale_minmax(x: float, lo: float, hi: float) -> float:
    x = clamp(x, lo, hi)
    return 0.0 if hi==lo else 100.0 * (x - lo) / (hi - lo)

def ratio_to_pct(r: float) -> float:
    return clamp(100.0 * r, 0.0, 100.0)

def invert_pct(p: float) -> float:
    return clamp(100.0 - p, 0.0, 100.0)

def normalize_metrics(spiky: Dict) -> Dict[str, float]:
    """Returns a flat dict of normalized metrics 0..100 following the config keys."""
    out = {}
    lang = spiky.get("language", {})
    vocal = spiky.get("vocal", {})
    facial = spiky.get("facial", {})
    inter = spiky.get("interaction", {})
    hlev = spiky.get("highlevel", {})

    # Language
    if "positivity" in lang: out["language.positivity"] = clamp(lang["positivity"], 0, 100)
    if "objectivity" in lang: out["language.objectivity"] = clamp(lang["objectivity"], 0, 100)
    if "proficiency" in lang: out["language.proficiency"] = clamp(lang["proficiency"], 0, 100)  # assume 0-100 mapped from A1..C1 upstream
    if "question_ratio" in lang: out["language.question_ratio"] = ratio_to_pct(lang["question_ratio"])
    if "offensiveness" in lang: out["language.offensiveness_inv"] = invert_pct(clamp(lang["offensiveness"],0,100))
    if "filler_ratio" in lang: out["language.filler_inv"] = invert_pct(ratio_to_pct(lang["filler_ratio"]))
    if "hedging_ratio" in lang: out["language.hedging_inv"] = invert_pct(ratio_to_pct(lang["hedging_ratio"]))
    if "avg_sentence_len" in lang: out["language.avg_sentence_len"] = scale_minmax(lang["avg_sentence_len"], 4, 25)  # words
    if "patience" in lang: out["language.patience"] = scale_minmax(lang["patience"], 0, 180)  # seconds

    # Vocal
    if "energy" in vocal: out["vocal.energy"] = clamp(vocal["energy"], 0, 100)
    if "emo_pos" in vocal: out["vocal.emo_pos"] = clamp(vocal["emo_pos"], 0, 100)
    if "emo_neu" in vocal: out["vocal.emo_neu"] = clamp(vocal["emo_neu"], 0, 100)
    if "emo_neg" in vocal: out["vocal.emo_neg_inv"] = invert_pct(clamp(vocal["emo_neg"],0,100))

    # Facial (optional)
    if "emo_pos" in facial: out["facial.emo_pos"] = clamp(facial["emo_pos"], 0, 100)
    if "emo_neu" in facial: out["facial.emo_neu"] = clamp(facial["emo_neu"], 0, 100)
    if "emo_dis" in facial: out["facial.emo_dis_inv"] = invert_pct(clamp(facial["emo_dis"],0,100))
    if "attention_att" in facial: out["facial.attentive"] = clamp(facial["attention_att"], 0, 100)
    if "attention_dist" in facial: out["facial.distracted_inv"] = invert_pct(clamp(facial["attention_dist"],0,100))

    # Interaction & derived
    if "talk_listen" in inter:
        # Expect ratio as speaker_share (0..1) or balance score 0..1
        # Convert to "balance" where 0.5 is best → score = 100 - |ratio - 0.5|*200
        r = max(0.0, min(1.0, inter["talk_listen"]))
        out["interaction.talk_listen_bal"] = clamp(100.0 - abs(r - 0.5) * 200.0, 0.0, 100.0)
    if "speed_wpm" in inter:
        # Optimal ~140 wpm; reasonable 90..180
        out["interaction.speed_wpm"] = scale_minmax(inter["speed_wpm"], 90, 180)

    if "action_items" in hlev:
        # Cap at 5 for scoring
        out["derived.action_items"] = scale_minmax(hlev["action_items"], 0, 5)
    if "followup_questions" in hlev:
        out["derived.followup_questions"] = scale_minmax(hlev["followup_questions"], 0, 10)

    return out

def apply_weights(norm: Dict[str,float]) -> Dict[str, float]:
    """Compute base factor scores from normalized metrics and weight matrix."""
    totals = {"precision":0.0, "resolve":0.0, "innovation":0.0, "harmony":0.0}
    for metric, s in norm.items():
        if metric not in WEIGHTS:
            # Unknown metric => ignore gracefully
            continue
        row = WEIGHTS[metric]
        row_sum = sum(row.values())
        if row_sum <= 0: 
            continue
        for f in totals.keys():
            totals[f] += (row[f] / row_sum) * s
    return totals

def multipliers_from_pattern(pattern: List[str]) -> Dict[str, float]:
    # pattern is ordered high→low priority, e.g. ["Resolve","Precision","Harmony","Innovation"]
    mp = {}
    order = ["primary","secondary","tertiary","quaternary"]
    for i, factor in enumerate(pattern):
        key = order[i] if i < 4 else "quaternary"
        mp[factor.lower()] = MULTS[key]
    # Ensure all present
    for f in ["precision","resolve","innovation","harmony"]:
        mp.setdefault(f, MULTS["quaternary"])
    return mp

def apply_pattern(base: Dict[str,float], pattern: List[str]) -> Dict[str,float]:
    mults = multipliers_from_pattern(pattern)
    out = {}
    for f, val in base.items():
        out[f] = min(100.0, val * mults[f])  # cap at 100
    return out

def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def alignment_score(factors: Dict[str,float], pattern: List[str]) -> float:
    # Target vector from multipliers order
    mults = multipliers_from_pattern(pattern)
    t = [mults["precision"], mults["resolve"], mults["innovation"], mults["harmony"]]
    s = [factors["precision"], factors["resolve"], factors["innovation"], factors["harmony"]]
    cos = cosine_similarity(s, t)
    return round(100.0 * cos, 2)

def drivers(norm: Dict[str,float], pattern: List[str]) -> List[Dict]:
    """Return per-metric contributions for explainability."""
    mults = multipliers_from_pattern(pattern)
    out = []
    for metric, s in norm.items():
        if metric not in WEIGHTS: 
            continue
        row = WEIGHTS[metric]
        row_sum = sum(row.values())
        for f, w in row.items():
            contrib = (w/row_sum) * s * mults[f]
            out.append({"metric": metric, "factor": f, "contribution": round(contrib,2)})
    out.sort(key=lambda d: d["contribution"], reverse=True)
    return out

def score_session(spiky_metrics: Dict, bcat_pattern_or_id) -> Dict:
    norm = normalize_metrics(spiky_metrics)
    base = apply_weights(norm)
    pattern = resolve_pattern(bcat_pattern_or_id)
    if not pattern:
        raise ValueError('Invalid BCAT pattern or id')
    fact = apply_pattern(base, pattern)
    align = alignment_score(fact, pattern)
    drv = drivers(norm, bcat_pattern)[:10]
    return {
        "factors": {k: round(v,2) for k,v in fact.items()},
        "alignment_pct": align,
        "top_drivers": drv,
        "normalized_metrics": {k: round(v,2) for k,v in norm.items()}
    }
