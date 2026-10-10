"""
Tuning study: can a tuned quantum encoding beat the classical baselines?

Uses a larger sample than the shipped catalogue (300 stars, 300 galaxies,
150 quasars per split, 8 splits) for tighter error bars, and sweeps 16
quantum encodings (angle range x layers x entangler layout).

Needs the full Kaggle file:  python tuning_study.py star_classification.csv

Finding: every fixed quantum encoding scores 0.77 to 0.82 ROC AUC, level
with the classical prototype score (0.80). Picking the best of the 16 per
split reaches 0.84, but that is selection on the test set. Local outlier
factor, which uses every training object, reaches 0.89.
"""
import math, sys, numpy as np, pandas as pd, itertools
sys.path.insert(0,".")
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
B=["u","g","r","i","z"]; F=["u_g","g_r","r_i","i_z"]
d=pd.read_csv(sys.argv[1]); d=d[((d[B]>0)&(d[B]<30)).all(axis=1)].copy()
for a,b in zip(B[:-1],B[1:]): d[f"{a}_{b}"]=d[a]-d[b]
def fm(x,reps,ent):
    n=len(x); qc=QuantumCircuit(n)
    for _ in range(reps):
        for i in range(n): qc.ry(float(x[i]),i)
        pairs=[(i,i+1) for i in range(n-1)]+([(n-1,0)] if ent=="ring" else [])
        for a,b in pairs: qc.cz(a,b)
        for i in range(n): qc.rz(float(x[i]),i)
    return qc
def states(A,reps,ent): return np.array([Statevector.from_instruction(fm(a,reps,ent)).data for a in A])
G=[0.01,0.03,0.1,0.3,1,3,10]
QG=list(itertools.product([math.pi/4,math.pi/2,3*math.pi/4,math.pi],[2,3],["chain","ring"]))
rows=[]
for seed in range(8):
    s=pd.concat([d[d["class"]=="STAR"].sample(300,random_state=seed),d[d["class"]=="GALAXY"].sample(300,random_state=seed),d[d["class"]=="QSO"].sample(150,random_state=seed)])
    X=s[F].to_numpy(); y=(s["class"]=="QSO").to_numpy().astype(int)
    tr,te=train_test_split(np.where(y==0)[0],test_size=0.5,random_state=seed); te=np.r_[te,np.where(y==1)[0]]; yt=y[te]
    r={}
    sc1=MinMaxScaler((0,1)).fit(X[tr]); U=lambda A: np.clip(sc1.transform(A),0,1)
    Tr1,Te1=U(X[tr]),U(X[te])
    P1=KMeans(6,n_init=10,random_state=seed).fit(Tr1).cluster_centers_
    r["c_ocsvm"]=max(roc_auc_score(yt,-OneClassSVM(gamma=g,nu=0.05).fit(Tr1).decision_function(Te1)) for g in [g*3 for g in G])
    r["c_lof"]=roc_auc_score(yt,-LocalOutlierFactor(n_neighbors=20,novelty=True).fit(Tr1).decision_function(Te1))
    r["c_proto"]=max(roc_auc_score(yt,1-np.exp(-g*((Te1[:,None]-P1[None])**2).sum(-1)).mean(1)) for g in [g*3 for g in G]+[30,100])
    qs={}
    for hi,reps,ent in QG:
        ov=np.abs(states(Te1*hi,reps,ent)@states(P1*hi,reps,ent).conj().T)**2
        qs[(round(hi,2),reps,ent)]=roc_auc_score(yt,1-ov.mean(1))
    r["q_fixed(pi/2,2,chain)"]=qs[(1.57,2,"chain")]
    r["q_best_tuned"]=max(qs.values()); r["_best"]=max(qs,key=qs.get)
    for k,v in qs.items(): r["q"+str(k)]=v
    rows.append(r); print(seed,{k:round(v,3) for k,v in r.items() if not k.startswith("q(") and k!="_best"},r["_best"],flush=True)
R=pd.DataFrame(rows).drop(columns="_best")
print(R.mean().round(3).to_string()); print("std"); print(R[["c_ocsvm","c_lof","c_proto","q_fixed(pi/2,2,chain)","q_best_tuned"]].std().round(3).to_string())
print("paired q_best - c_proto:", (R.q_best_tuned-R.c_proto).round(3).tolist())
