
from typing import Dict, List, Tuple
import json, os
import numpy as np

FACTORS = ["Precision","Resolve","Innovation","Harmony"]

def _load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

CONF_DIR = os.path.join(os.path.dirname(__file__), "config")
WEIGHTS   = _load_json(os.path.join(CONF_DIR, "weights.json"))
MULT      = _load_json(os.path.join(CONF_DIR, "multipliers.json"))
PATTERNS  = _load_json(os.path.join(CONF_DIR, "patterns.json"))

def load_patterns_by(pattern_id: int = None, pattern_name: str = None, bcat_pattern: List[str] = None) -> List[str]:
    if bcat_pattern:
        assert set(bcat_pattern) == set(FACTORS)
        return bcat_pattern
    if pattern_id is not None:
        for p in PATTERNS:
            if p["id"] == pattern_id:
                return p["order"]
        raise ValueError("Unknown pattern_id")
    if pattern_name is not None:
        for p in PATTERNS:
            if p["name"] == pattern_name:
                return p["order"]
        raise ValueError("Unknown pattern_name")
    return ["Resolve","Precision","Innovation","Harmony"]

# ---------- helpers ----------
def clamp01(x): return max(0.0, min(1.0, float(x)))
def to100(x):  return clamp01(x) * 100.0
def minmax_to100(x, lo, hi):
    x = float(x)
    if x <= lo: return 0.0
    if x >= hi: return 100.0
    return (x - lo) / (hi - lo) * 100.0
def inv100(x): return 100.0 - float(x)
def talk_balance_from_ratio(r): return 100.0 - abs(clamp01(r) - 0.5) * 200.0

def normalize_metrics(spiky: Dict) -> Dict[str, float]:
    L = spiky.get("language",{}) or {}
    V = spiky.get("vocal",{}) or {}
    F = spiky.get("facial",{}) or {}
    I = spiky.get("interaction",{}) or {}
    H = spiky.get("highlevel",{}) or {}

    out: Dict[str, float] = {}

    # --- New: handle class-level positivity/objectivity/energy & strings ---
    # Positivity classes: {positive, neutral, negative}
    pos_classes = L.get("positivity_classes") or L.get("polarity") or (L.get("positivity") if isinstance(L.get("positivity"), dict) else None)
    if isinstance(pos_classes, dict):
        p   = float(pos_classes.get("positive", 0))
        neu = float(pos_classes.get("neutral", 0))
        neg = float(pos_classes.get("negative", 0))
        # conservative: treat neutral as half-positive
        L.setdefault("positivity", max(0.0, min(1.0, p + 0.5*neu)))
        # explicit negativity (higher = healthier / less negative). Accepts 0–1 or 0–100.
        out["negativity_inv"] = 100.0 - (neg * 100.0 if neg <= 1 else neg)

    # Objectivity classes: {objective, subjective} OR string "objective"/"subjective"
    obj_classes = L.get("objectivity_classes") or (L.get("objectivity") if isinstance(L.get("objectivity"), dict) else None)
    if isinstance(obj_classes, dict):
        if "objective" in obj_classes:
            L.setdefault("objectivity", float(obj_classes.get("objective", 0)))
        elif "subjective" in obj_classes:
            L.setdefault("objectivity", 1.0 - float(obj_classes.get("subjective", 0)))
    elif isinstance(L.get("objectivity"), str):
        L["objectivity"] = 1.0 if str(L["objectivity"]).lower().startswith("obj") else 0.0

    # Question flag may come as string label
    if isinstance(L.get("question"), str):
        L["question"] = 1.0 if L["question"].lower().startswith("question") else 0.0

    # Offensiveness may come as string label
    if isinstance(L.get("offensiveness"), str):
        L["offensiveness"] = 1.0 if L["offensiveness"].lower().startswith("offen") else 0.0

    # -------- Language numeric fields --------
    if "positivity" in L: out["positivity"] = to100(L["positivity"] if L["positivity"] <= 1 else L["positivity"]/100)
    if "objectivity" in L: out["objectivity"] = to100(L["objectivity"] if L["objectivity"] <= 1 else L["objectivity"]/100)
    if "proficiency" in L: out["proficiency"] = to100(L["proficiency"] if L["proficiency"] <= 1 else L["proficiency"]/100)

    # CEFR mapping if provided
    if "proficiency_cefr" in L:
        cefr = str(L.get("proficiency_cefr")).upper()
        cefr_map = {"A1":0.2,"A2":0.4,"B1":0.6,"B2":0.8,"C1":1.0}
        L.setdefault("proficiency", cefr_map.get(cefr, 0.6))

    if "hedging_ratio" in L:   out["hedging_inv"]   = inv100(to100(L["hedging_ratio"]))
    if "filler_ratio" in L:    out["filler_inv"]    = inv100(to100(L["filler_ratio"]))
    if "question_ratio" in L:  out["question_ratio"]= to100(L["question_ratio"])

    if "avg_sentence_len" in L: out["avg_sentence_len_norm"] = minmax_to100(L["avg_sentence_len"], 4, 25)
    if "offensiveness" in L:    out["offensiveness_inv"] = inv100(to100(L["offensiveness"]))
    if "patience" in L:         out["patience_norm"] = minmax_to100(L["patience"], 0, 180)

    # Keywords aggregate (optional)
    kw = L.get("keywords") or {}
    if isinstance(kw, dict) and len(kw) > 0:
        vals = []
        for _,v in kw.items():
            v100 = to100(v if v <= 1 else v/100)
            vals.append(v100)
        out["kw_strength"] = float(np.mean(vals)) if vals else 0.0

    # Language Emotions (13)
    em = L.get("emotions") or {}
    def _to100(v): return to100(v if v <= 1 else v/100)
    emo_keys = [
        "nervousness","disapproval","sadness","anger","confusion",
        "neutral","curiosity","thoughtful","admiration","optimism",
        "approval","joy","excitement"
    ]
    for ek in emo_keys:
        if ek in em:
            val100 = _to100(em[ek])
            key = f"lang_emo_{ek}"
            if ek in ["nervousness","disapproval","sadness","anger","confusion"]:
                val100 = 100.0 - val100
            out[key] = val100

    # -------- Vocal --------
    # Energy classes (vocal): energy may be a dict {"energetic": x, "monotonic": y}
    if isinstance(V.get("energy"), dict):
        ener = float(V["energy"].get("energetic", 0.0))
        mono = float(V["energy"].get("monotonic", 0.0))
        V["energy"] = ener if ener > 0 else (1.0 - mono)

    # Accept aggregate OR class-level breakdown
    ve = V.get("emotions") or {}
    if isinstance(ve, dict) and ve:
        if "happy" in ve:      V.setdefault("emo_pos", ve.get("happy"))
        if "neutral" in ve:    V.setdefault("emo_neu", ve.get("neutral"))
        if ("sad" in ve) or ("angry" in ve):
            V.setdefault("emo_neg", (ve.get("sad",0)+ve.get("angry",0)))
    if "energy" in V:     out["energy"]      = to100(V["energy"] if V["energy"] <= 1 else V["energy"]/100)
    if "emo_pos" in V:    out["emo_pos"]     = to100(V["emo_pos"] if V["emo_pos"] <= 1 else V["emo_pos"]/100)
    if "emo_neu" in V:    out["emo_neu"]     = to100(V["emo_neu"] if V["emo_neu"] <= 1 else V["emo_neu"]/100)
    if "emo_neg" in V:    out["emo_neg_inv"] = inv100(to100(V["emo_neg"]))

    # -------- Facial (optional) --------
    fe = F.get("emotions") or {}
    if isinstance(fe, dict) and fe:
        if "happy" in fe:         F.setdefault("emo_pos", fe.get("happy"))
        if "neutral" in fe:       F.setdefault("emo_neu", fe.get("neutral"))
        if "surprised" in fe:     F["emo_neu"] = float(F.get("emo_neu",0.0)) + float(fe.get("surprised",0.0))
        if ("dissatisfied" in fe) or ("annoyed" in fe):
            F.setdefault("emo_dis", (fe.get("dissatisfied",0)+fe.get("annoyed",0)))
    # Attention breakdown accepted, map to att / dist
    att = F.get("attention") or {}
    if isinstance(att, dict) and att:
        if "attentive" in att:  F.setdefault("attention_att", att.get("attentive"))
        if "distracted" in att: F.setdefault("attention_dist", att.get("distracted"))
        # include "normal" as half-credit towards attention
        if "normal" in att:
            F["attention_att"] = float(F.get("attention_att",0.0)) + 0.5*float(att.get("normal",0.0))
    if "attention_att" in F: out["attention_att"] = to100(F["attention_att"] if F["attention_att"] <= 1 else F["attention_att"]/100)
    if "attention_dist" in F: out["attention_dist_inv"] = inv100(to100(F["attention_dist"]))
    if "emo_pos" in F:     out["facial_emo_pos"]   = to100(F["emo_pos"] if F["emo_pos"] <= 1 else F["emo_pos"]/100)
    if "emo_neu" in F:     out["facial_emo_neu"]   = to100(F["emo_neu"] if F["emo_neu"] <= 1 else F["emo_neu"]/100)
    if "emo_dis" in F:     out["facial_emo_dis_inv"]= inv100(to100(F["emo_dis"]))

    # -------- Interaction --------
    if "talk_listen" in I: out["talk_balance"]   = talk_balance_from_ratio(float(I["talk_listen"]))
    if "speed_wpm"  in I:  out["speed_wpm_norm"] = minmax_to100(I["speed_wpm"], 90, 180)

    # -------- High-level --------
    if "action_items" in H:        out["action_items"] = to100(H["action_items"] if H["action_items"] <= 1 else H["action_items"]/100)
    if "followup_questions" in H:  out["followup_questions"] = to100(H["followup_questions"] if H["followup_questions"] <= 1 else H["followup_questions"]/100)

    return out

def base_factors(s_norm: Dict[str, float]) -> Dict[str, float]:
    base = {f: 0.0 for f in FACTORS}
    for m, s in s_norm.items():
        if m not in WEIGHTS: 
            continue
        row = WEIGHTS[m]
        denom = sum(row.get(f,0.0) for f in FACTORS) or 1.0
        for f in FACTORS:
            share = row.get(f,0.0) / denom
            base[f] += s * share
    for f in FACTORS:
        base[f] = float(max(0.0, min(100.0, base[f])))
    return base

def apply_multipliers(base: Dict[str, float], pattern: List[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
    pos = {pattern[0]:"primary", pattern[1]:"secondary", pattern[2]:"tertiary", pattern[3]:"quaternary"}
    pf = {f: MULT[pos[f]] for f in FACTORS}
    scored = {f: float(min(100.0, base[f] * pf[f])) for f in FACTORS}
    return scored, pf

def alignment_percent(scored: Dict[str, float], pf: Dict[str,float]) -> float:
    s = np.array([scored[f] for f in FACTORS], dtype=float)
    t = np.array([pf[f] for f in FACTORS], dtype=float)
    t = t / (np.linalg.norm(t) + 1e-9)
    denom = (np.linalg.norm(s) + 1e-9) * (np.linalg.norm(t) + 1e-9)
    cos = float(np.dot(s, t) / denom) if denom > 0 else 0.0
    return float(max(0.0, min(100.0, cos * 100.0)))

def top_drivers(s_norm: Dict[str,float], pf: Dict[str,float], k: int = 10):
    contribs = []
    for m, s in s_norm.items():
        if m not in WEIGHTS: 
            continue
        row = WEIGHTS[m]
        denom = sum(row.get(f,0.0) for f in FACTORS) or 1.0
        for f in FACTORS:
            share = row.get(f,0.0) / denom
            contribs.append({
                "metric": m,
                "factor": f,
                "contribution": float(s * share * pf[f])
            })
    contribs.sort(key=lambda x: x["contribution"], reverse=True)
    return contribs[:k]

def score_payload(spiky: Dict, pattern_order: List[str]) -> Dict:
    s_norm = normalize_metrics(spiky)
    base = base_factors(s_norm)
    scored, pf = apply_multipliers(base, pattern_order)
    align = alignment_percent(scored, pf)
    drivers = top_drivers(s_norm, pf, k=10)
    return {
        "factors": {
            "precision": scored["Precision"],
            "resolve": scored["Resolve"],
            "innovation": scored["Innovation"],
            "harmony": scored["Harmony"]
        },
        "alignment_pct": align,
        "top_drivers": drivers,
        "normalized_metrics": s_norm
    }
