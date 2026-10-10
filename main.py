"""
Qollab entry point: score ONE telescope object for "is this unusual?" on the
selected backend.

Project files needed: main.py, quantum_core.py, model.json
Qollab creates `backend` before this code starts (Select QPU dialog), and
calls main() itself after this file has loaded. The parameters of main()
appear as fields under Run settings:

  Object Number   which held-out test object to score.
                  0 to 95 are normal (stars and galaxies), 96 to 135 are quasars.
  Shots           how many times the circuit is measured (about 45 ms each on
                  the IonQ noisy simulator, plus about 12 s of fixed overhead).
  Layers          encoding layers. 2 is the design used for every reported
                  result; other values are for studying noise against depth.
  Chart           draw a bar chart of the overlaps after the numbers.

On an IonQ backend the run is one 24-qubit circuit, because IonQ accepts one
circuit per run. On any other backend the same six blocks are sent as six
4-qubit circuits, which local simulators accept and handle easily.
"""

import json
import time

from qiskit import transpile

from quantum_core import (N_PROTOTYPES, anomaly_score, block_circuits,
                          combined_circuit, exact_overlaps, mitigate,
                          overlaps_from_block_counts, overlaps_from_counts,
                          survival_factor)

DESIGN_LAYERS = 2


def is_ionq(be):
    """True for IonQ backends, which accept exactly one circuit per run."""
    text = f"{type(be).__module__} {type(be).__name__} {getattr(be, 'name', '')}"
    return "ionq" in text.lower()


def run_counts(be, circuit, shots):
    """Run one circuit. Local simulators may need it rewritten in their own
    gate set first; IonQ compiles the circuit itself."""
    try:
        return be.run(circuit, shots=shots).result().get_counts()
    except Exception:
        if is_ionq(be):
            raise                      # never send a second job to IonQ
        return be.run(transpile(circuit, be), shots=shots).result().get_counts()


def draw_chart(measured, corrected, exact, cal, title):
    """Grouped bars: measured, corrected and exact overlap for each block."""
    import matplotlib.pyplot as plt

    ink, muted, surface = "#0b0b0b", "#52514e", "#fcfcfb"
    series = [("Measured", "#2a78d6", list(measured) + [cal[0]]),
              ("Corrected", "#eb6834", list(corrected) + [cal[1]]),
              ("Exact (noise-free)", "#1baf7a", list(exact) + [cal[2]])]
    labels = [f"Prototype {i + 1}" for i in range(len(measured))] + ["Calibration"]
    width = 0.24

    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=surface)
    ax.set_facecolor(surface)
    for k, (name, colour, values) in enumerate(series):
        xs = [i + (k - 1) * (width + 0.02) for i in range(len(labels))]
        ax.bar(xs, values, width, label=name, color=colour, zorder=2)
        for x, v in zip(xs, values):
            ax.text(x, v + 0.015, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=7.5, color=muted)
    ax.axvline(len(measured) - 0.5, color="#c9c8c2", linewidth=1, zorder=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, color=ink, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="y", colors=muted, labelsize=8, length=0)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Overlap (1 = identical)", color=muted, fontsize=9)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c2")
    ax.set_title(title, color=ink, fontsize=11, loc="left", pad=28)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=ink, ncol=3,
              loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0.2)
    fig.tight_layout()
    plt.show()


def main(object_number: int = 0, shots: int = 400, layers: int = DESIGN_LAYERS, chart: bool = True):
    with open("model.json") as f:          # literal path: required by Qollab
        model = json.load(f)

    layers = max(1, min(4, layers))
    objects = model["test_objects"]
    obj = objects[object_number % len(objects)]
    x, prototypes = obj["x"], model["prototypes"]
    cal_a, cal_b = model["cal_a"], model["cal_b"]
    threshold = model["threshold"]

    print(f"object {object_number} of {len(objects)} | colours (scaled): {x}")
    t0 = time.time()
    if is_ionq(backend):  # noqa: F821
        qc = combined_circuit(x, prototypes, cal_a, cal_b, layers)
        print(f"one circuit: {qc.num_qubits} qubits, depth {qc.depth()}, "
              f"{qc.count_ops().get('cz', 0)} CZ gates, {N_PROTOTYPES} prototype "
              f"comparisons + 1 calibration block, {layers} layers, {shots} shots")
        blocks = overlaps_from_counts(run_counts(backend, qc, shots))  # noqa: F821
    else:
        circuits = block_circuits(x, prototypes, cal_a, cal_b, layers)
        print(f"not an IonQ backend: sending the six blocks as {len(circuits)} "
              f"separate {circuits[0].num_qubits}-qubit circuits, {layers} layers, "
              f"{shots} shots each")
        blocks = overlaps_from_block_counts(
            [run_counts(backend, c, shots) for c in circuits])  # noqa: F821
    print(f"run time: {time.time() - t0:.1f}s")

    measured, cal_measured = blocks[:N_PROTOTYPES], blocks[N_PROTOTYPES]
    cal_exact = exact_overlaps(cal_a, [cal_b], layers)[0]
    f_noise = survival_factor(cal_measured, cal_exact)
    corrected = mitigate(measured, f_noise)
    exact = exact_overlaps(x, prototypes, layers)

    print(f"calibration block: measured {cal_measured:.3f}, noise-free value "
          f"{cal_exact:.3f} -> {f_noise:.0%} of the signal survived the noise")
    if f_noise >= 0.99:
        print("  (no noise detected on this backend, so nothing needed correcting; pick "
              "an IonQ simulator under Select QPU to see the correction at work)")
    print("overlap with each normal prototype")
    print("  measured  :", [round(v, 2) for v in measured])
    print("  corrected :", [round(v, 2) for v in corrected])
    print("  exact     :", [round(v, 2) for v in exact])
    raw, score, exact_score = (anomaly_score(measured), anomaly_score(corrected),
                               anomaly_score(exact))
    truth = obj["class"] + (" (quasar = anomaly)" if obj["class"] == "QSO" else " (normal)")
    if layers == DESIGN_LAYERS:
        verdict = "UNUSUAL - worth a look" if score > threshold else "ordinary"
        print(f"anomaly score: {raw:.3f} measured | {score:.3f} corrected | "
              f"{exact_score:.3f} exact | flag above {threshold:.3f}")
        print("verdict :", verdict)
    else:
        verdict = "no verdict"
        print(f"anomaly score: {raw:.3f} measured | {score:.3f} corrected | "
              f"{exact_score:.3f} exact")
        print(f"verdict : not given, the threshold was set for {DESIGN_LAYERS} layers")
    print("truth   :", truth)

    if chart:
        try:
            draw_chart(measured, corrected, exact,
                       (cal_measured, mitigate([cal_measured], f_noise)[0], cal_exact),
                       f"Object {object_number}: {verdict.split(' - ')[0].lower()} "
                       f"(score {score:.2f}, truth: {obj['class']})")
        except Exception as err:           # a chart problem must not fail the run
            print(f"(chart skipped: {err})")
