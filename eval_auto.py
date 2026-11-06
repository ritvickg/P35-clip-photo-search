"""Automatic evaluation. A photo counts as relevant to a query if its human-written Flickr captions contain the rule words
(| = any of, + = all of). Reports hit@5, precision@5 and latency for raw / template / expanded."""
import json, time
import pandas as pd
import core
EMB, NAMES = core.load_index()
CAPS = {k: " ".join(v).lower() for k, v in json.load(open("assets/captions.json")).items()}
Q = pd.read_csv("eval/queries_auto.csv")

def relevant(name, rule):
    t = CAPS[name]
    return all(any(w in t for w in part.split("|")) for part in rule.split("+"))

rows, fails = [], {}
for mode in ["raw", "template", "expanded"]:
    hit = prec = lat = used = 0
    for q, rule in Q.values:
        if not any(relevant(n, rule) for n in NAMES):   # no relevant photo exists -> cannot be scored
            continue
        t = time.time(); top = [n for n, _ in core.search(core.text_query_vec(q, mode), EMB, NAMES, 5)]
        lat += (time.time() - t) * 1000; used += 1
        r = [relevant(n, rule) for n in top]
        hit += any(r); prec += sum(r) / 5
        if not any(r): fails.setdefault(mode, []).append(q)
    rows.append({"mode": mode, "queries_scored": used, "hit@5": round(hit / used, 3), "precision@5": round(prec / used, 3), "avg_ms": round(lat / used, 1)})
df = pd.DataFrame(rows); print(df.to_string(index=False)); df.to_csv("eval/results.csv", index=False)
print("\nFailure cases (no relevant photo in top 5):", fails)
