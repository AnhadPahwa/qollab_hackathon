"""
Local check of the calibration block under gate-level noise.

Runs the real 24-qubit circuit for 22 test objects on a local simulator with
depolarizing noise on every gate (1.2% per CZ, 0.1% per rotation), then
compares the raw and the calibrated anomaly scores with the exact ones.
This is an independent check: the correction formula assumes a simple noise
model, and here the noise is applied gate by gate.

Needs qiskit-aer:  pip install qiskit-aer ;  python noise_check.py
"""

import json

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from quantum_core import (N_PROTOTYPES, anomaly_score, combined_circuit,
                          exact_overlaps, mitigate, overlaps_from_counts,
                          survival_factor)

SHOTS = 400
model = json.load(open("model.json"))
objects, protos = model["test_objects"], model["prototypes"]
cal_a, cal_b = model["cal_a"], model["cal_b"]
cal_exact = exact_overlaps(cal_a, [cal_b])[0]

noise = NoiseModel()
noise.add_all_qubit_quantum_error(depolarizing_error(0.012, 2), ["cz"])
noise.add_all_qubit_quantum_error(depolarizing_error(0.001, 1), ["ry", "rz"])
sim = AerSimulator(method="matrix_product_state", noise_model=noise,
                   basis_gates=["ry", "rz", "cz"])

raw, fixed, exact = [], [], []
for i in list(range(0, 96, 8)) + list(range(96, 136, 4)):
    qc = combined_circuit(objects[i]["x"], protos, cal_a, cal_b)
    counts = sim.run(transpile(qc, sim, optimization_level=0), shots=SHOTS).result().get_counts()
    blocks = overlaps_from_counts(counts)
    f = survival_factor(blocks[N_PROTOTYPES], cal_exact)
    raw.append(anomaly_score(blocks[:N_PROTOTYPES]))
    fixed.append(anomaly_score(mitigate(blocks[:N_PROTOTYPES], f)))
    exact.append(objects[i]["exact_score"])

raw, fixed, exact = map(np.array, (raw, fixed, exact))
print(f"{len(raw)} objects, {SHOTS} shots each")
print(f"mean error of raw score        : {abs(raw - exact).mean():.3f} (bias {np.mean(raw - exact):+.3f})")
print(f"mean error of calibrated score : {abs(fixed - exact).mean():.3f} (bias {np.mean(fixed - exact):+.3f})")
