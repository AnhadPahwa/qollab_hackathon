"""
Offline comparison: classical baselines vs the quantum scorer, run on the
local (noise-free) simulator. Qollab's IonQ backend takes one circuit per
run, so this many-circuit study cannot run there; its results ship as files.

Reads   objects.csv
Writes  model.json    scaler, prototypes, threshold and the test objects,
                      everything main.py needs to score one object on IonQ
        results.json  the comparison table and the shot-count study

Method (same for every model)
  - "Normal" = stars and galaxies, anomaly = quasars. Labels score only.
  - Train on 60% of the normal objects; test on the other 40% plus all quasars.
  - Features scaled to [0, pi/2] using the training normals only.
  - Five prototypes of normal (k-means centres of the training normals).
  - Repeated over several random splits; the mean and spread are reported.
"""

import json
import math

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM

from quantum_core import N_FEATURES, N_PROTOTYPES, state

FEATURES = ["u_g", "g_r", "r_i", "i_z"]
HI = math.pi / 2
SEEDS = list(range(10))
SHOT_GRID = [40, 100, 400, 1000]
F_NOISE = 0.89         # signal survival measured on the IonQ Forte 1 noisy simulator
CAL_SHIFT = 0.10       # calibration pair: prototype 0 and a copy shifted by this much
FLOOR = 1 / 2 ** N_FEATURES
MODEL_SEED = 0          # the split that model.json is built from
GAMMAS = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]

df = pd.read_csv("objects.csv")
X = df[FEATURES].to_numpy(float)
y = (df["class"] == "QSO").astype(int).to_numpy()
cls = df["class"].to_numpy()


def split(seed):
    idx_n, idx_a = np.where(y == 0)[0], np.where(y == 1)[0]
    tr, te = train_test_split(idx_n, test_size=0.4, random_state=seed)
    te = np.r_[te, idx_a]
    sc = MinMaxScaler(feature_range=(0, HI)).fit(X[tr])
    f = lambda A: np.clip(sc.transform(A), 0, HI)
    return tr, te, sc, f(X[tr]), f(X[te]), y[te]


def states(A):
    return np.array([state(a) for a in A])


rows, shot_rows, model = [], {s: [] for s in SHOT_GRID}, None
noise_rows = []
for seed in SEEDS:
    tr, te, sc, Tr, Te, yt = split(seed)
    rng = np.random.default_rng(seed)
    r = {}

    # --- classical -------------------------------------------------------
    # gamma chosen by test AUC: this favours the classical side on purpose.
    r["One-class SVM, RBF kernel (tuned)"] = max(
        roc_auc_score(yt, -OneClassSVM(gamma=g, nu=0.05).fit(Tr).decision_function(Te))
        for g in GAMMAS)
    r["Isolation forest"] = roc_auc_score(
        yt, -IsolationForest(n_estimators=300, random_state=seed).fit(Tr).decision_function(Te))
    r["Local outlier factor"] = roc_auc_score(
        yt, -LocalOutlierFactor(n_neighbors=20, novelty=True).fit(Tr).decision_function(Te))

    # five prototypes of "normal": k-means centres of the training normals
    P = KMeans(N_PROTOTYPES, n_init=10, random_state=seed).fit(Tr).cluster_centers_
    r["Classical prototype score, RBF (tuned)"] = max(
        roc_auc_score(yt, 1 - np.exp(-g * ((Te[:, None] - P[None]) ** 2).sum(-1)).mean(1))
        for g in GAMMAS)

    # --- quantum (noise-free statevector) ---------------------------------
    s_tr, s_te, s_p = states(Tr), states(Te), states(P)
    K_tr = np.abs(s_tr @ s_tr.conj().T) ** 2
    K_te = np.abs(s_te @ s_tr.conj().T) ** 2
    r["Quantum kernel one-class SVM"] = roc_auc_score(
        yt, -OneClassSVM(kernel="precomputed", nu=0.05).fit(K_tr).decision_function(K_te))
    ov = np.abs(s_te @ s_p.conj().T) ** 2          # test objects x prototypes
    r["Quantum prototype score (exact)"] = roc_auc_score(yt, 1 - ov.mean(1))
    rows.append(r)

    # --- shot-count study: what the one-circuit IonQ run can resolve -------
    for shots in SHOT_GRID:
        aucs = []
        for _ in range(20):
            est = rng.binomial(shots, np.clip(ov, 0, 1)) / shots
            aucs.append(roc_auc_score(yt, 1 - est.mean(1)))
        shot_rows[shots].append(float(np.mean(aucs)))

    # --- noise study: what the measured noise does, and what calibration fixes
    # Each block returns P = f * exact + (1 - f) / 16, sampled with `shots` shots.
    ov_tr = np.abs(s_tr @ s_p.conj().T) ** 2
    thr = float(np.percentile(1 - ov_tr.mean(1), 80))
    cal_a = P[0]
    cal_b = np.where(P[0] + CAL_SHIFT <= HI, P[0] + CAL_SHIFT, P[0] - CAL_SHIFT)
    cal_exact = float(abs(np.vdot(state(cal_b), state(cal_a))) ** 2)
    for shots in (400,):
        for _ in range(20):
            noisy = rng.binomial(shots, F_NOISE * ov + (1 - F_NOISE) * FLOOR) / shots
            cal = rng.binomial(shots, F_NOISE * cal_exact + (1 - F_NOISE) * FLOOR,
                               size=len(ov)) / shots
            f_hat = np.clip((cal - FLOOR) / (cal_exact - FLOOR), 0.3, 1.0)[:, None]
            fixed = np.clip((noisy - (1 - f_hat) * FLOOR) / f_hat, 0, 1)
            clean = rng.binomial(shots, np.clip(ov, 0, 1)) / shots
            for name, est in (("No noise", clean), ("IonQ-level noise, uncorrected", noisy),
                              ("IonQ-level noise, calibrated", fixed)):
                sc_ = 1 - est.mean(1)
                noise_rows.append({"case": name, "auc": roc_auc_score(yt, sc_),
                                   "false_alarms": float((sc_[yt == 0] > thr).mean()),
                                   "quasars_flagged": float((sc_[yt == 1] > thr).mean())})

    if seed == MODEL_SEED:
        model = {
            "features": FEATURES, "scale_max": HI,
            "data_min": sc.data_min_.tolist(), "data_max": sc.data_max_.tolist(),
            "prototypes": np.round(P, 5).tolist(),
            "cal_a": np.round(cal_a, 5).tolist(), "cal_b": np.round(cal_b, 5).tolist(),
            "threshold": round(thr, 4),
            "threshold_rule": "80th percentile of the exact score on training normals (about 1 in 5 normals flagged)",
            "test_objects": [
                {"id": int(i), "class": str(cls[i]), "x": np.round(Te[k], 5).tolist(),
                 "exact_score": round(float(1 - ov[k].mean()), 4)}
                for k, i in enumerate(te)],
        }

res = pd.DataFrame(rows)
table = [{"model": c, "auc_mean": round(float(res[c].mean()), 3),
          "auc_std": round(float(res[c].std()), 3)} for c in res.columns]
nz = pd.DataFrame(noise_rows).groupby("case", sort=False).mean()
noise_table = [{"case": k, "auc": round(float(v.auc), 3),
                "false_alarm_rate": round(float(v.false_alarms), 3),
                "quasars_flagged": round(float(v.quasars_flagged), 3)}
               for k, v in nz.iterrows()]
shots_table = [{"shots": s, "samples_per_object": s * N_PROTOTYPES,
                "auc_mean": round(float(np.mean(v)), 3)} for s, v in shot_rows.items()]

print(f"{len(df)} objects, {int(y.sum())} quasars, {len(SEEDS)} random splits\n")
print(f"{'model':42s} {'ROC AUC':>8s} {'+/-':>6s}")
for t in table:
    print(f"{t['model']:42s} {t['auc_mean']:8.3f} {t['auc_std']:6.3f}")
print("\nQuantum prototype score with a limited shot budget (no gate noise):")
for t in shots_table:
    print(f"  {t['shots']:4d} shots ({t['samples_per_object']:4d} samples/object)  "
          f"AUC {t['auc_mean']:.3f}")

print(f"\nEffect of noise at 400 shots (signal survival f = {F_NOISE}), "
      "threshold set without noise:")
print(f"  {'case':32s} {'AUC':>6s} {'false alarms':>13s} {'quasars flagged':>16s}")
for t in noise_table:
    print(f"  {t['case']:32s} {t['auc']:6.3f} {t['false_alarm_rate']:13.1%} "
          f"{t['quasars_flagged']:16.1%}")

json.dump(model, open("model.json", "w"))
json.dump({"n_objects": len(df), "n_anomalies": int(y.sum()), "n_splits": len(SEEDS),
           "comparison": table, "shot_study": shots_table,
           "noise_study": {"signal_survival": F_NOISE, "shots": 400, "rows": noise_table}}, open("results.json", "w"), indent=1)
print("\nwrote model.json and results.json")
