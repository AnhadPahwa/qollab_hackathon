"""
Build the small labelled catalogue the project ships with.

Input : star_classification.csv  (Kaggle "Stellar Classification Dataset -
        SDSS17", 100,000 SDSS DR17 objects with u, g, r, i, z magnitudes and
        a spectroscopic class of GALAXY, STAR or QSO).
Output: objects.csv  (N_NORMAL ordinary objects + N_ANOMALY quasars)

Features are four colour indices (differences of neighbouring magnitudes):
    u_g, g_r, r_i, i_z
Colours describe an object's spectrum shape independent of its brightness,
and four of them map onto four qubits.

"Normal" = stars and galaxies, in equal numbers. "Anomaly" = quasars (QSO).
The class column is used only to score the detectors, never to train them.

Usage:  python prepare_data.py [path/to/star_classification.csv]
"""

import sys

import pandas as pd

SEED = 7
N_NORMAL = 240      # half stars, half galaxies
N_ANOMALY = 40
BANDS = ["u", "g", "r", "i", "z"]
FEATURES = ["u_g", "g_r", "r_i", "i_z"]

path = sys.argv[1] if len(sys.argv) > 1 else "star_classification.csv"
df = pd.read_csv(path)
n0 = len(df)

# The catalogue marks missing photometry with -9999; drop those rows, and
# keep magnitudes inside the survey's sensible range.
ok = (df[BANDS] > 0).all(axis=1) & (df[BANDS] < 30).all(axis=1)
df = df[ok].copy()
print(f"{n0} rows read, {n0 - len(df)} dropped for bad photometry")

for a, b in zip(BANDS[:-1], BANDS[1:]):
    df[f"{a}_{b}"] = df[a] - df[b]

parts = [
    df[df["class"] == "STAR"].sample(N_NORMAL // 2, random_state=SEED),
    df[df["class"] == "GALAXY"].sample(N_NORMAL // 2, random_state=SEED),
    df[df["class"] == "QSO"].sample(N_ANOMALY, random_state=SEED),
]
out = pd.concat(parts).sample(frac=1, random_state=SEED)  # shuffle
out = out[FEATURES + ["redshift", "class"]].reset_index(drop=True)
out[FEATURES] = out[FEATURES].round(5)
out.to_csv("objects.csv", index=False)
print(out["class"].value_counts().to_string())
print(f"wrote objects.csv ({len(out)} rows)")
