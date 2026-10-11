"""
Backend helpers shared by main.py and experiments.py.

IonQ backends on Qollab accept exactly ONE circuit per run. Every other
backend (the built-in simulator, the IBM noise-model simulators) runs in the
browser and accepts as many circuits as the code sends.
"""

from qiskit import QuantumCircuit, transpile

from quantum_core import N_FEATURES


def is_ionq(be):
    """True for IonQ backends, which accept exactly one circuit per run."""
    text = f"{type(be).__module__} {type(be).__name__} {getattr(be, 'name', '')}"
    return "ionq" in text.lower()


def backend_label(be):
    name = getattr(be, "name", None)
    name = name() if callable(name) else name
    return str(name or type(be).__name__)


class Runner:
    """Runs circuits on one backend and returns counts."""

    def __init__(self, be):
        self.be = be
        self.ionq = is_ionq(be)
        self.needs_transpile = False
        self.jobs = 0

    def counts(self, circuit, shots):
        """Counts for one circuit. Local simulators may need the circuit
        rewritten in their own gate set first; IonQ compiles it itself."""
        self.jobs += 1
        if not self.needs_transpile:
            try:
                return self.be.run(circuit, shots=shots).result().get_counts()
            except Exception:
                if self.ionq:
                    raise                  # never send a second job to IonQ
                self.needs_transpile = True
        return self.be.run(transpile(circuit, self.be), shots=shots).result().get_counts()

    def block_zero_fractions(self, blocks, shots):
        """P(0000) for each 4-qubit block circuit (no measurements attached).

        On IonQ the blocks are placed side by side in ONE circuit. Elsewhere
        each block is sent as its own small circuit.
        """
        n = N_FEATURES
        zero = "0" * n
        if self.ionq:
            width = n * len(blocks)
            qc = QuantumCircuit(width, width)
            for b, block in enumerate(blocks):
                qc.compose(block, qubits=list(range(b * n, (b + 1) * n)), inplace=True)
            qc.measure(range(width), range(width))
            counts = self.counts(qc, shots)
            total = sum(counts.values())
            hits = [0] * len(blocks)
            for bits, c in counts.items():
                bits = bits.replace(" ", "")
                for b in range(len(blocks)):
                    hi = width - b * n     # qubit 0 is the rightmost character
                    if bits[hi - n:hi] == zero:
                        hits[b] += c
            return [h / total for h in hits]
        out = []
        for block in blocks:
            qc = QuantumCircuit(n, n)
            qc.compose(block, inplace=True)
            qc.measure(range(n), range(n))
            counts = self.counts(qc, shots)
            total = sum(counts.values())
            out.append(sum(c for bits, c in counts.items()
                           if bits.replace(" ", "") == zero) / total)
        return out
