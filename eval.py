"""U3 evaluation: top-5 hit rate on 15 queries, comparing raw / template / expanded modes.

Step 1:  python eval.py make     -> writes eval/review.csv (query, file, relevant blank)
Step 2:  open eval/review.csv, put 1 (relevant) or 0 in the 'relevant' column for every row
Step 3:  python eval.py score    -> prints hit@5 and precision@5 per mode + average latency
Hit@5 = share of queries with at least one relevant photo in the top 5.
"""
import sys, time
import pandas as pd
import core

EMB, NAMES = core.load_index()
MODES = ["raw", "template", "expanded"]
Q = pd.read_csv("eval/queries.csv")["query"].tolist()


def ranked(mode):
    res, lat = {}, []
    for q in Q:
        t = time.time()
        res[q] = [n for n, _ in core.search(core.text_query_vec(q, mode), EMB, NAMES, 5)]
        lat.append((time.time() - t) * 1000)
    return res, sum(lat) / len(lat)


if sys.argv[1] == "make":
    rows = set()
    for m in MODES:
        r, _ = ranked(m)
        rows |= {(q, f) for q, fs in r.items() for f in fs}
    pd.DataFrame(sorted(rows), columns=["query", "file", ]).assign(relevant="").to_csv("eval/review.csv", index=False)
    print("Label eval/review.csv (1/0) then run: python eval.py score")
else:
    rel = pd.read_csv("eval/review.csv")
    rel = {(a, b): int(c) for a, b, c in rel[["query", "file", "relevant"]].values}
    out = []
    for m in MODES:
        r, lat = ranked(m)
        hit = sum(any(rel[(q, f)] for f in fs) for q, fs in r.items()) / len(Q)
        prec = sum(sum(rel[(q, f)] for f in fs) / 5 for q, fs in r.items()) / len(Q)
        out.append({"mode": m, "hit@5": round(hit, 3), "precision@5": round(prec, 3), "avg_ms": round(lat, 1)})
    df = pd.DataFrame(out); print(df.to_string(index=False)); df.to_csv("eval/results.csv", index=False)
    # failure cases: queries with zero hits in the best mode
    best = max(out, key=lambda d: d["hit@5"])["mode"]
    r, _ = ranked(best)
    print("\nFailure cases (", best, "):", [q for q, fs in r.items() if not any(rel[(q, f)] for f in fs)])
