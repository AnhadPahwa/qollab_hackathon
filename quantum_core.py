"""
Quantum scoring core for the telescope anomaly project.

One object is scored in ONE 24-qubit circuit (IonQ on Qollab accepts one
circuit per run), built from six independent 4-qubit blocks.

Blocks 0-4 compare the candidate with five prototypes of "normal": each
applies the encoding of the candidate, then the inverse encoding of a
prototype. The fraction of shots where the block reads 0000 estimates the
overlap |<phi(prototype)|phi(candidate)>|^2.

Block 5 is a calibration block. It compares two fixed reference points whose
overlap is known exactly, so the shortfall in its measured value shows how
much the backend's noise shrank the overlaps in this same run. That factor is
used to correct blocks 0-4 (see `mitigate`).

Anomaly score = 1 - mean corrected overlap across the five prototype blocks.

Needs only numpy and core Qiskit.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

N_FEATURES = 4
N_BLOCKS = 6        # blocks in the circuit
N_PROTOTYPES = 5    # blocks 0-4; block 5 is the calibration block
N_QUBITS = N_FEATURES * N_BLOCKS
REPS = 2   # encoding layers. Use >= 2: with one layer the CZ gates cancel in
           # the overlap and the kernel is a product of single-qubit terms.


def feature_map(x, reps=REPS):
    """Data re-uploading encoding: RY(x) rotations, a CZ chain, RZ(x) phases,
    repeated `reps` times. Features are expected in [0, pi/2]."""
    n = len(x)
    qc = QuantumCircuit(n)
    for _ in range(reps):
        for i in range(n):
            qc.ry(float(x[i]), i)
        for i in range(n - 1):
            qc.cz(i, i + 1)
        for i in range(n):
            qc.rz(float(x[i]), i)
    return qc


def overlap_block(a, b, reps=REPS):
    """4-qubit circuit whose P(0000) is the overlap of the encodings of a and b."""
    qc = QuantumCircuit(N_FEATURES)
    qc.compose(feature_map(a, reps), inplace=True)
    qc.compose(feature_map(b, reps).inverse(), inplace=True)
    return qc


def block_pairs(candidate, prototypes, cal_a, cal_b):
    """The six comparisons a run makes: five prototypes, then calibration."""
    pairs = [(candidate, p) for p in prototypes] + [(cal_a, cal_b)]
    assert len(pairs) == N_BLOCKS
    return pairs


def combined_circuit(candidate, prototypes, cal_a, cal_b, reps=REPS):
    """The single 24-qubit job. Block b sits on qubits 4b .. 4b+3."""
    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    for b, (x, p) in enumerate(block_pairs(candidate, prototypes, cal_a, cal_b)):
        qubits = list(range(b * N_FEATURES, (b + 1) * N_FEATURES))
        qc.compose(overlap_block(x, p, reps), qubits=qubits, inplace=True)
    qc.measure(range(N_QUBITS), range(N_QUBITS))
    return qc


def block_circuits(candidate, prototypes, cal_a, cal_b, reps=REPS):
    """The same six blocks as six separate 4-qubit circuits, for backends
    that accept many jobs per run but may not handle 24 qubits."""
    out = []
    for x, p in block_pairs(candidate, prototypes, cal_a, cal_b):
        qc = QuantumCircuit(N_FEATURES, N_FEATURES)
        qc.compose(overlap_block(x, p, reps), inplace=True)
        qc.measure(range(N_FEATURES), range(N_FEATURES))
        out.append(qc)
    return out


def overlaps_from_counts(counts):
    """Per-block P(0000) from 24-bit counts (qubit 0 is the rightmost bit)."""
    total = sum(counts.values())
    zeros = [0] * N_BLOCKS
    for bits, c in counts.items():
        bits = bits.replace(" ", "")
        for b in range(N_BLOCKS):
            hi = N_QUBITS - b * N_FEATURES
            if bits[hi - N_FEATURES:hi] == "0" * N_FEATURES:
                zeros[b] += c
    return [z / total for z in zeros]


def overlaps_from_block_counts(counts_list):
    """Per-block P(0000) from a list of six 4-bit count dictionaries."""
    out = []
    for counts in counts_list:
        total = sum(counts.values())
        zero = sum(c for bits, c in counts.items()
                   if bits.replace(" ", "") == "0" * N_FEATURES)
        out.append(zero / total)
    return out


def state(x, reps=REPS):
    """Noise-free encoded state of one object (16 amplitudes)."""
    return Statevector.from_instruction(feature_map(x, reps)).data


def exact_overlaps(candidate, prototypes, reps=REPS):
    s = state(candidate, reps)
    return [float(abs(np.vdot(state(p, reps), s)) ** 2) for p in prototypes]


def survival_factor(cal_measured, cal_exact):
    """Fraction of the signal that survived the noise, from the calibration block.

    Noise model: with probability f a block behaves ideally, otherwise its
    4-bit outcome is uniformly random, so P(0000) = f * exact + (1 - f) / 16.
    """
    floor = 1.0 / 2 ** N_FEATURES
    f = (cal_measured - floor) / (cal_exact - floor)
    return float(min(1.0, max(0.3, f)))


def mitigate(overlaps, f):
    """Undo the shrinkage: invert P = f * exact + (1 - f) / 16 for each block."""
    floor = 1.0 / 2 ** N_FEATURES
    return [float(min(1.0, max(0.0, (m - (1 - f) * floor) / f))) for m in overlaps]


def anomaly_score(overlaps):
    return 1.0 - float(np.mean(overlaps))