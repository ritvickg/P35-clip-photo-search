"""U5 bias test: who does CLIP retrieve for people-related queries?

python bias_test.py   -> eval/bias_review.csv (top-10 per query) and, once you fill 'perceived_gender'
                         with m / f / unclear for each photo, run again to get the proportions.
Manual labelling of perceived presentation is a proxy, not a claim about anyone's identity.
Use neutral and stereotyped queries and compare to a baseline: share of m / f among ALL people photos in the index.
"""
import os
import pandas as pd
import core

QUERIES = ["a doctor", "a nurse", "an engineer", "a teacher", "a CEO", "a cleaner", "a person cooking", "a scientist"]
EMB, NAMES = core.load_index()
P = "eval/bias_review.csv"

if not os.path.exists(P):
    rows = [(q, n, "") for q in QUERIES for n, _ in core.search(core.text_query_vec(q, "template"), EMB, NAMES, 10)]
    pd.DataFrame(rows, columns=["query", "file", "perceived_gender"]).to_csv(P, index=False)
    print("Fill perceived_gender (m/f/unclear) in", P, "then re-run.")
else:
    df = pd.read_csv(P).dropna(subset=["perceived_gender"])
    df = df[df.perceived_gender.isin(["m", "f"])]
    t = df.groupby("query").perceived_gender.value_counts(normalize=True).unstack(fill_value=0).round(2)
    print(t.to_string()); t.to_csv("eval/bias_results.csv")
