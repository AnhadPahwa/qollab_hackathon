"""
Experiments behind the README's noise results. Chosen with the Experiment
field under Run settings (see main.py):

  depth    Noise against circuit depth, in ONE run. Six calibration blocks
           with 1 to 6 encoding layers sit side by side. Each has a known
           noise-free answer, so each reports how much signal survived at
           that depth. Works on any backend.

  sample   Scores many test objects and compares the verdicts with and
           without the calibration correction. Needs a backend that accepts
           many circuits per run, so it runs on the browser simulators (for
           example an IBM noise-model simulator). On IonQ, which accepts one
           circuit per run, it scores a single object.

Every run ends with one line starting "RESULT " that holds the numbers as
JSON. collect_runs.py turns pasted console output into tables.
"""

import json
import time

from quantum_core import (N_PROTOTYPES, anomaly_score, block_pairs,
                          exact_overlaps, mitigate, overlap_block,
                          survival_factor)
from runner import backend_label

INK, MUTED, SURFACE, RULE = "#0b0b0b", "#52514e", "#fcfcfb", "#c9c8c2"
BLUE, ORANGE = "#2a78d6", "#eb6834"
DEPTHS = [1, 2, 3, 4, 5, 6]


def emit(record):
    print("RESULT " + json.dumps(record, separators=(",", ":")))


def _axes(title, xlabel, ylabel, size=(7.5, 4.2)):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=size, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(color="#e6e5e0", linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=12)
    return plt, fig, ax


def score_object(runner, model, index, shots, layers=2):
    """Score one test object. Returns a dict of everything measured."""
    obj = model["test_objects"][index % len(model["test_objects"])]
    protos, cal_a, cal_b = model["prototypes"], model["cal_a"], model["cal_b"]
    blocks = [overlap_block(a, b, layers)
              for a, b in block_pairs(obj["x"], protos, cal_a, cal_b)]
    got = runner.block_zero_fractions(blocks, shots)
    measured, cal_measured = got[:N_PROTOTYPES], got[N_PROTOTYPES]
    cal_exact = exact_overlaps(cal_a, [cal_b], layers)[0]
    f = survival_factor(cal_measured, cal_exact)
    corrected = mitigate(measured, f)
    exact = exact_overlaps(obj["x"], protos, layers)
    return {"object": index, "class": obj["class"], "shots": shots, "layers": layers,
            "survival": round(f, 4), "cal_measured": round(cal_measured, 4),
            "cal_exact": round(cal_exact, 4),
            "measured": [round(v, 4) for v in measured],
            "corrected": [round(v, 4) for v in corrected],
            "exact": [round(v, 4) for v in exact],
            "score_measured": round(anomaly_score(measured), 4),
            "score_corrected": round(anomaly_score(corrected), 4),
            "score_exact": round(anomaly_score(exact), 4)}


def depth_study(runner, model, shots, chart=True):
    """Signal survival at 1 to 6 encoding layers, from one run."""
    cal_a, cal_b = model["cal_a"], model["cal_b"]
    blocks = [overlap_block(cal_a, cal_b, d) for d in DEPTHS]
    exact = [exact_overlaps(cal_a, [cal_b], d)[0] for d in DEPTHS]
    cz = [b.count_ops().get("cz", 0) for b in blocks]
    how = "one 24-qubit circuit" if runner.ionq else "six 4-qubit circuits"
    print(f"depth study on {backend_label(runner.be)}: {how}, {shots} shots")
    t0 = time.time()
    measured = runner.block_zero_fractions(blocks, shots)
    wall = time.time() - t0
    survival = [survival_factor(m, e) for m, e in zip(measured, exact)]
    print(f"run time: {wall:.1f}s")
    print("layers  CZ gates  noise-free  measured  signal survived")
    for d, g, e, m, f in zip(DEPTHS, cz, exact, measured, survival):
        print(f"{d:6d} {g:9d} {e:11.3f} {m:9.3f} {f:15.0%}")
    print("(1 layer: the CZ gates cancel in an overlap test, so a compiler may "
          "remove them and that row can look better than its gate count suggests)")
    emit({"experiment": "depth", "backend": backend_label(runner.be), "shots": shots,
          "wall_s": round(wall, 1), "layers": DEPTHS, "cz_gates": cz,
          "exact": [round(v, 4) for v in exact],
          "measured": [round(v, 4) for v in measured],
          "survival": [round(v, 4) for v in survival]})
    if chart:
        try:
            plt, fig, ax = _axes(f"Signal surviving the noise, by encoding depth "
                                 f"({backend_label(runner.be)})",
                                 "Encoding layers", "Signal survived")
            ax.plot(DEPTHS, survival, color=BLUE, linewidth=2, marker="o",
                    markersize=7, zorder=3)
            for d, f in zip(DEPTHS, survival):
                ax.annotate(f"{f:.0%}", (d, f), textcoords="offset points",
                            xytext=(0, 9), ha="center", fontsize=8, color=MUTED)
            ax.set_ylim(0, 1.1)
            ax.set_xticks(DEPTHS)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
            fig.tight_layout()
            plt.show()
        except Exception as err:
            print(f"(chart skipped: {err})")


def sample_indices(model, size):
    """Evenly spaced test objects: half normal, half quasars."""
    objs = model["test_objects"]
    normal = [i for i, o in enumerate(objs) if o["class"] != "QSO"]
    quasar = [i for i, o in enumerate(objs) if o["class"] == "QSO"]
    half = max(1, size // 2)

    def spread(ix, k):
        k = min(k, len(ix))
        return [ix[(j * len(ix)) // k] for j in range(k)]
    return spread(normal, half) + spread(quasar, half)


def sample_study(runner, model, shots, size, object_number=0, chart=True):
    """Verdicts with and without the correction, over many objects."""
    thr = model["threshold"]
    if runner.ionq:
        print("IonQ accepts one circuit per run, so this run scores one object. "
              "Run it once per object (objects to use: "
              f"{sample_indices(model, size)}), or pick a browser simulator to "
              "run the whole sample at once.")
        indices = [object_number]
    else:
        indices = sample_indices(model, size)
    print(f"sample study on {backend_label(runner.be)}: {len(indices)} objects, "
          f"{shots} shots, flag above {thr:.3f}")
    print("object class   survived  exact  measured  corrected   "
          "verdict(measured -> corrected)  truth-match")
    t0 = time.time()
    rows = []
    for i in indices:
        r = score_object(runner, model, i, shots)
        rows.append(r)
        anomaly = r["class"] == "QSO"
        vm, vc = r["score_measured"] > thr, r["score_corrected"] > thr
        word = lambda v: "unusual " if v else "ordinary"
        print(f"{i:6d} {r['class']:7s} {r['survival']:8.0%} {r['score_exact']:6.3f} "
              f"{r['score_measured']:9.3f} {r['score_corrected']:10.3f}   "
              f"{word(vm)} -> {word(vc)}            {'yes' if vc == anomaly else 'no'}")
    wall = time.time() - t0

    n = len(rows)
    normal = [r for r in rows if r["class"] != "QSO"]
    quasar = [r for r in rows if r["class"] == "QSO"]
    rate = lambda rs, key: round(sum(r[key] > thr for r in rs) / len(rs), 4) if rs else None
    err = lambda key: sum(abs(r[key] - r["score_exact"]) for r in rows) / n
    summary = {
        "experiment": "sample", "backend": backend_label(runner.be), "shots": shots,
        "objects": n, "normal": len(normal), "quasars": len(quasar),
        "wall_s": round(wall, 1), "threshold": thr,
        "mean_survival": round(sum(r["survival"] for r in rows) / n, 4),
        "mean_abs_error_measured": round(err("score_measured"), 4),
        "mean_abs_error_corrected": round(err("score_corrected"), 4),
        "false_alarm_rate": {k: rate(normal, f"score_{k}")
                             for k in ("exact", "measured", "corrected")},
        "quasars_flagged": {k: rate(quasar, f"score_{k}")
                            for k in ("exact", "measured", "corrected")},
        "verdicts_matching_noise_free": {
            k: sum((r[f"score_{k}"] > thr) == (r["score_exact"] > thr) for r in rows)
            for k in ("measured", "corrected")},
        "rows": [{k: r[k] for k in ("object", "class", "survival", "score_exact",
                                    "score_measured", "score_corrected")} for r in rows],
    }
    print(f"\nrun time: {wall:.1f}s for {runner.jobs} circuits")
    if n > 1:
        s = summary
        print(f"signal survived, average       : {s['mean_survival']:.0%}")
        print(f"score error, uncorrected       : {s['mean_abs_error_measured']:.3f}")
        print(f"score error, corrected         : {s['mean_abs_error_corrected']:.3f}")
        print("                                 noise-free  uncorrected  corrected")
        for label, key in (("normal objects flagged", "false_alarm_rate"),
                           ("quasars flagged", "quasars_flagged")):
            v = s[key]
            if v["exact"] is None:
                continue
            print(f"{label:31s}: {v['exact']:9.0%} {v['measured']:12.0%} "
                  f"{v['corrected']:10.0%}")
        v = s["verdicts_matching_noise_free"]
        print(f"verdicts matching noise-free   : uncorrected {v['measured']}/{n}, "
              f"corrected {v['corrected']}/{n}")
    emit(summary)

    if chart and n > 1:
        try:
            plt, fig, ax = _axes(f"Measured score against the noise-free score "
                                 f"({backend_label(runner.be)}, {n} objects)",
                                 "Noise-free anomaly score", "Score from the run",
                                 size=(6.4, 5.2))
            xs = [r["score_exact"] for r in rows]
            lo = min(xs + [r["score_corrected"] for r in rows]) - 0.03
            hi = max(xs + [r["score_measured"] for r in rows]) + 0.03
            ax.plot([lo, hi], [lo, hi], color=RULE, linewidth=1.2, zorder=1)
            ax.axhline(thr, color=MUTED, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
            ax.text(hi, thr, " flag above", va="bottom", ha="right", fontsize=8, color=MUTED)
            ax.scatter(xs, [r["score_measured"] for r in rows], s=46, color=BLUE,
                       edgecolor=SURFACE, linewidth=1.2, label="Uncorrected", zorder=3)
            ax.scatter(xs, [r["score_corrected"] for r in rows], s=46, color=ORANGE,
                       marker="s", edgecolor=SURFACE, linewidth=1.2,
                       label="Corrected", zorder=3)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left")
            fig.tight_layout()
            plt.show()
        except Exception as err:
            print(f"(chart skipped: {err})")
