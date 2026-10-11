"""
Turn pasted Qollab console output into tables.

Every experiment run prints one line starting "RESULT ". Paste the console
output of one or more runs into a text file, then:

    python collect_runs.py pasted_output.txt

It prints Markdown tables for the README and adds the runs to
ionq_runs.json under "experiment_runs" (duplicates are skipped).
"""

import json
import sys


def load_results(text):
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("RESULT "):
            try:
                out.append(json.loads(line[len("RESULT "):]))
            except ValueError:
                print("skipped a RESULT line that was cut off or edited")
    return out


def pct(v):
    return "n/a" if v is None else f"{v:.0%}"


def depth_tables(runs):
    for r in runs:
        print(f"\nDepth study, {r['backend']}, {r['shots']} shots\n")
        print("| Encoding layers | CZ gates per block | Noise-free value | Measured | Signal survived |")
        print("|---|---|---|---|---|")
        for d, g, e, m, f in zip(r["layers"], r["cz_gates"], r["exact"],
                                 r["measured"], r["survival"]):
            print(f"| {d} | {g} | {e:.3f} | {m:.3f} | {f:.0%} |")


def sample_tables(runs):
    by_backend = {}
    for r in runs:
        by_backend.setdefault((r["backend"], r["shots"]), []).extend(r["rows"])
    for (backend, shots), rows in by_backend.items():
        thr = runs[0]["threshold"]
        seen, uniq = set(), []
        for row in rows:                       # one row per object
            if row["object"] not in seen:
                seen.add(row["object"])
                uniq.append(row)
        rows = sorted(uniq, key=lambda x: x["object"])
        n = len(rows)
        normal = [x for x in rows if x["class"] != "QSO"]
        quasar = [x for x in rows if x["class"] == "QSO"]
        rate = lambda rs, k: (sum(x[k] > thr for x in rs) / len(rs)) if rs else None
        err = lambda k: sum(abs(x[k] - x["score_exact"]) for x in rows) / n
        match = lambda k: sum((x[k] > thr) == (x["score_exact"] > thr) for x in rows)
        print(f"\nSample study, {backend}, {shots} shots: {n} objects "
              f"({len(normal)} normal, {len(quasar)} quasars), threshold {thr:.3f}\n")
        print("| | Noise-free | Uncorrected | Corrected |")
        print("|---|---|---|---|")
        print(f"| Normal objects flagged | {pct(rate(normal, 'score_exact'))} | "
              f"{pct(rate(normal, 'score_measured'))} | {pct(rate(normal, 'score_corrected'))} |")
        print(f"| Quasars flagged | {pct(rate(quasar, 'score_exact'))} | "
              f"{pct(rate(quasar, 'score_measured'))} | {pct(rate(quasar, 'score_corrected'))} |")
        print(f"| Verdicts matching noise-free | {n} of {n} | {match('score_measured')} of {n} | "
              f"{match('score_corrected')} of {n} |")
        print(f"| Mean error of the score | 0 | {err('score_measured'):.3f} | "
              f"{err('score_corrected'):.3f} |")
        print(f"\nAverage signal survival: {sum(x['survival'] for x in rows) / n:.0%}")
        print("\n| Object | True class | Signal survived | Exact | Measured | Corrected |")
        print("|---|---|---|---|---|---|")
        for x in rows:
            print(f"| {x['object']} | {x['class']} | {x['survival']:.0%} | "
                  f"{x['score_exact']:.3f} | {x['score_measured']:.3f} | "
                  f"{x['score_corrected']:.3f} |")


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    results = load_results(open(sys.argv[1], encoding="utf-8").read())
    if not results:
        sys.exit("no RESULT lines found")
    depth_tables([r for r in results if r["experiment"] == "depth"])
    sample_tables([r for r in results if r["experiment"] == "sample"])

    try:
        store = json.load(open("ionq_runs.json"))
    except FileNotFoundError:
        store = {}
    kept = store.setdefault("experiment_runs", [])
    known = {json.dumps(r, sort_keys=True) for r in kept}
    new = [r for r in results if json.dumps(r, sort_keys=True) not in known]
    kept.extend(new)
    json.dump(store, open("ionq_runs.json", "w"), indent=1)
    print(f"\n{len(new)} new run(s) added to ionq_runs.json ({len(kept)} in total)")


if __name__ == "__main__":
    main()
