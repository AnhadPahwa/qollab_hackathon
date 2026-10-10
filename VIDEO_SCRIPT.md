# Demo video script (2 to 3 minutes)

Screen recording with voiceover. About 400 words, which is roughly 2 minutes 45 at a normal pace. A 400-shot run takes about 35 seconds, so cut the wait in editing.

## 0:00 to 0:25 — The problem

**Show:** the project page title, then the first rows of `objects.csv`.

> Sky surveys catalogue hundreds of millions of objects. Nearly all are ordinary stars and galaxies. The hard part is finding the few that aren't, without knowing in advance what you're looking for. This project scores one object for how unusual it is, using a single quantum circuit on the IonQ simulator, and compares that against classical methods on real survey data.

## 0:25 to 1:05 — How it works

**Show:** the six-block diagram in the README, then `combined_circuit` in `quantum_core.py`.

> Each object is four colours, so it fits on four qubits. To compare an object with a prototype of "normal", we encode the object, undo the encoding of the prototype, and count how often we get all zeros. That is the overlap between them.
>
> IonQ on Qollab accepts one circuit per run. So five comparisons sit side by side on 20 qubits, and the last four qubits run a calibration block: a comparison whose answer we already know. Whatever the noise does to that block, it did to the others, so we use it to correct them.

## 1:05 to 1:50 — Running it

**Show:** Select QPU with the IonQ Forte 1 simulator, the Run settings form, press Run with Object Number 0, then the console output. Repeat with a quasar.

> Here is a star. The calibration line says how much of the signal survived the noise. The corrected overlaps sit close to the exact values, the score is below the threshold, and the verdict is "ordinary", which matches its true class.
>
> And here is a quasar. [Read out the score and verdict you actually get.]

Use objects whose verdicts you have already checked. If a quasar is missed in your recording, say so: about a third are at this threshold.

## 1:50 to 2:30 — The results

**Show:** the two results tables at the top of the README.

> Against four classical detectors, over ten random splits, the quantum scorer reaches an AUC of 0.77. The best classical methods reach 0.80. So it comes close, but it does not win, and the classical side was tuned in its own favour.
>
> The calibration is what works. We measured that the simulator's noise shrinks every overlap by about eleven percent. Uncorrected, that flags more than half of all normal objects. With the calibration block, the false-alarm rate goes back to a quarter, where it should be.

## 2:30 to 2:50 — What we learned

**Show:** the "What we found" headings.

> Three things came out of this. On a one-circuit backend, you can carry your noise calibration inside the same circuit. A one-layer encoding quietly loses its entanglement in an overlap test. And on four features, there is no quantum advantage to claim. Everything here can be reproduced from the scripts in the project.
>
> This effort is supported by Qollab and IonQ.
