"""
Checks for quantum_core.py. No backend needed.

    python test_core.py        (or: pytest test_core.py)
"""

import json
import math

import numpy as np
from qiskit.quantum_info import Statevector

from quantum_core import (N_BLOCKS, N_FEATURES, N_PROTOTYPES, N_QUBITS,
                          anomaly_score, block_circuits, combined_circuit,
                          exact_overlaps, feature_map, mitigate, overlap_block,
                          overlaps_from_block_counts, overlaps_from_counts,
                          state, survival_factor)

rng = np.random.default_rng(0)
X = rng.uniform(0, math.pi / 2, size=(8, N_FEATURES))


def p0000(circuit):
    return float(abs(Statevector.from_instruction(circuit).data[0]) ** 2)


def test_object_overlaps_fully_with_itself():
    for x in X:
        assert abs(exact_overlaps(x, [x])[0] - 1.0) < 1e-9


def test_overlap_is_symmetric_and_bounded():
    a, b = X[0], X[1]
    ab, ba = exact_overlaps(a, [b])[0], exact_overlaps(b, [a])[0]
    assert abs(ab - ba) < 1e-9 and 0.0 <= ab <= 1.0


def test_block_circuit_matches_state_overlap():
    a, b = X[2], X[3]
    assert abs(p0000(overlap_block(a, b)) - exact_overlaps(a, [b])[0]) < 1e-9


def test_one_layer_has_no_entanglement_effect():
    """With one layer the CZ gates cancel: the overlap equals the product of
    single-qubit overlaps. With two layers it does not."""
    a, b = X[4], X[5]
    product = 1.0
    for i in range(N_FEATURES):
        product *= exact_overlaps([a[i]], [[b[i]]], reps=1)[0]
    assert abs(exact_overlaps(a, [b], reps=1)[0] - product) < 1e-9
    product2 = 1.0
    for i in range(N_FEATURES):
        product2 *= exact_overlaps([a[i]], [[b[i]]], reps=2)[0]
    assert abs(exact_overlaps(a, [b], reps=2)[0] - product2) > 1e-3


def test_combined_circuit_shape():
    qc = combined_circuit(X[0], X[1:6], X[6], X[7])
    assert qc.num_qubits == N_QUBITS == 24
    assert qc.count_ops()["measure"] == N_QUBITS
    # blocks are independent: no gate touches qubits from two blocks
    for inst in qc.data:
        blocks = {qc.find_bit(q).index // N_FEATURES for q in inst.qubits}
        assert len(blocks) == 1


def test_counts_parsing_uses_the_right_bits():
    # every block reads 0000 except block 2 (qubits 8-11), which reads 0001
    bits = ["0"] * N_QUBITS
    bits[N_QUBITS - 1 - 8] = "1"           # qubit 8 is 9th from the right
    got = overlaps_from_counts({"".join(bits): 10, "0" * N_QUBITS: 30})
    assert got == [1.0, 1.0, 0.75, 1.0, 1.0, 1.0]


def test_combined_and_separate_circuits_agree():
    """The 24-qubit circuit and the six 4-qubit circuits give the same exact
    block probabilities."""
    cand, protos, ca, cb = X[0], X[1:6], X[6], X[7]
    separate = block_circuits(cand, protos, ca, cb)
    assert len(separate) == N_BLOCKS
    expect = exact_overlaps(cand, protos) + exact_overlaps(ca, [cb])
    for qc, e in zip(separate, expect):
        qc = qc.remove_final_measurements(inplace=False)
        assert abs(p0000(qc) - e) < 1e-9
    assert overlaps_from_block_counts([{"0000": 3, "0101": 1}] * N_BLOCKS) == [0.75] * N_BLOCKS


def test_mitigation_is_identity_without_noise():
    assert survival_factor(0.9, 0.9) == 1.0
    assert mitigate([0.2, 0.7], 1.0) == [0.2, 0.7]


def test_mitigation_inverts_the_noise_model():
    f, floor = 0.85, 1 / 16
    exact = np.array([0.95, 0.6, 0.1])
    noisy = f * exact + (1 - f) * floor
    cal_exact = 0.938
    f_hat = survival_factor(f * cal_exact + (1 - f) * floor, cal_exact)
    assert abs(f_hat - f) < 1e-9
    assert np.allclose(mitigate(list(noisy), f_hat), exact)


def test_model_file_is_consistent():
    m = json.load(open("model.json"))
    assert len(m["prototypes"]) == N_PROTOTYPES
    assert all(len(p) == N_FEATURES for p in m["prototypes"])
    hi = m["scale_max"] + 1e-4      # values are stored to 5 decimals
    for o in m["test_objects"]:
        assert all(0 <= v <= hi for v in o["x"])
        score = anomaly_score(exact_overlaps(o["x"], m["prototypes"]))
        assert abs(score - o["exact_score"]) < 1e-3
    cal = exact_overlaps(m["cal_a"], [m["cal_b"]])[0]
    assert 0.85 < cal < 0.99           # high, but not trivially 1


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        fn()
        print("ok  ", name)
    print(f"{len(tests)} tests passed")
