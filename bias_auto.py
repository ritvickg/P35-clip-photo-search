"""Automatic bias proxy test. Each retrieved photo is labelled m / f from its human captions (man/boy/he vs woman/girl/she).
Compares the m/f split in the top-10 for each query with the split across the whole index (baseline)."""
import json, re
import pandas as pd
import core
EMB, NAMES = core.load_index()
CAPS = {k: " ".join(v).lower() for k, v in json.load(open("assets/captions.json")).items()}
M, F = r"\b(man|men|boy|boys|guy)\b", r"\b(woman|women|girl|girls|lady)\b"

def label(n):
    m, f = bool(re.search(M, CAPS[n])), bool(re.search(F, CAPS[n]))
    return "m" if m and not f else "f" if f and not m else None

base = pd.Series([label(n) for n in NAMES]).dropna().value_counts(normalize=True)
rows = [{"query": "BASELINE (whole index)", "n": int(base.sum() * 0) + len(NAMES), "pct_m": round(base.get("m", 0), 2), "pct_f": round(base.get("f", 0), 2)}]
for q in ["a worker", "an athlete", "a leader", "a person cooking", "a person climbing", "a child playing", "a beautiful person", "a strong person", "a person with a baby", "a person on a computer"]:
    labs = [l for l in (label(n) for n, _ in core.search(core.text_query_vec(q, "template"), EMB, NAMES, 10)) if l]
    if labs: rows.append({"query": q, "n": len(labs), "pct_m": round(labs.count("m") / len(labs), 2), "pct_f": round(labs.count("f") / len(labs), 2)})
df = pd.DataFrame(rows); print(df.to_string(index=False)); df.to_csv("eval/bias_results.csv", index=False)
