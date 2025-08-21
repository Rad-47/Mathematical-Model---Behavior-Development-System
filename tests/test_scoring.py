import json, importlib.util, os
spec = importlib.util.spec_from_file_location("scorer", os.path.join("scorer.py"))
sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)

def test_smoke():
    req = json.load(open("examples/example_request.json"))
    res = sc.score_payload(req["spiky"], req["bcat_pattern"])
    assert "factors" in res and "alignment_pct" in res
    for v in res["factors"].values():
        assert 0.0 <= v <= 100.0
    assert 0.0 <= res["alignment_pct"] <= 100.0
