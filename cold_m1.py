#!/usr/bin/env python3
"""
Is M2 standing in for a cold lower layer that the thermal class cannot represent?
For a reference run and any number of variants (lower-layer floor extended, Clapeyron 660, ...), report
  - how often the best M1 model of a cap sits at the lower-layer temperature floor, by class of the cap;
  - the M1-class posterior mass at the floor, by class;
  - for each variant: what the reference M2 caps become, and the lower-layer temperature of the best M1 model
    in the reference M2 caps (does it go below 1400 K when allowed to?).
    python cold_m1.py REF_DIR VARIANT_DIR [VARIANT_DIR ...]
"""
import sys, glob, json, numpy as np
from invert_bins2 import parse


def load(d):
    J = {}
    for f in sorted(glob.glob(f"{d}/*.json")):
        j = json.load(open(f)); j["cls"] = 2 if j["lnB21"] > 3 else (1 if j["lnB10"] > 3 else 0)
        j["P2"] = np.exp(j["classes"]["M2"]["logZ"]) / sum(np.exp(j["classes"][c]["logZ"]) for c in ("M0", "M1", "M2"))
        J[j["bin"]] = j
    return J


def floor_stats(J, label):
    T2best = {c: [] for c in (0, 1, 2)}; mass = {c: [] for c in (0, 1, 2)}; T2m2 = []
    allT2 = sorted({float(k) for j in J.values() for k in j["classes"]["M1"]["post_T2"]}); floor = allT2[0]
    for j in J.values():
        b = parse(j["classes"]["M1"]["best"]); T2best[j["cls"]].append(b["T2"])
        mass[j["cls"]].append(j["classes"]["M1"]["post_T2"].get(str(floor), j["classes"]["M1"]["post_T2"].get(floor, 0.0)))
        if j["cls"] == 2: T2m2.append(parse(j["classes"]["M2"]["best"])["T2"])
    n = len(J); fr = [100 * np.mean([j["cls"] == c for j in J.values()]) for c in (0, 1, 2)]
    print(f"\n{label}: {n} caps, M0/M1/M2 = {fr[0]:.0f}/{fr[1]:.0f}/{fr[2]:.0f} %; lower-layer floor in this run: {floor:.0f} K")
    print(f"{'cap class':10s} {'n':>4s}  best-M1 T2 at floor  M1 posterior mass at floor (mean)  median best-M1 T2")
    for c, lab in ((0, "M0"), (1, "M1"), (2, "M2")):
        t = np.array(T2best[c]); m = np.array(mass[c])
        if len(t): print(f"{lab:10s} {len(t):4d}  {100 * np.mean(t == floor):17.0f} %  {100 * m.mean():31.0f} %  {np.median(t):16.0f} K")
    if T2m2: print(f"best M2 model in the M2 caps: lower-layer T2 median {np.median(T2m2):.0f} K, at floor in {100 * np.mean(np.array(T2m2) == floor):.0f} %")
    return floor


ref = load(sys.argv[1]); floor_stats(ref, sys.argv[1])
m2ref = [b for b, j in ref.items() if j["cls"] == 2]
for d in sys.argv[2:]:
    V = load(d); fl = floor_stats(V, d)
    common = [b for b in ref if b in V]
    same = np.mean([ref[b]["cls"] == V[b]["cls"] for b in common]) * 100
    r = np.corrcoef([ref[b]["P2"] for b in common], [V[b]["P2"] for b in common])[0, 1]
    now = [V[b]["cls"] for b in m2ref if b in V]
    print(f"vs reference: same class {same:.0f} %, r(P(M2)) {r:.2f}; the {len(now)} reference M2 caps are now M0/M1/M2 = "
          f"{now.count(0)}/{now.count(1)}/{now.count(2)}")
    tcold = [parse(V[b]["classes"]["M1"]["best"])["T2"] for b in m2ref if b in V]
    print(f"best M1 lower-layer T2 in the reference M2 caps: median {np.median(tcold):.0f} K, below 1400 K in {100 * np.mean(np.array(tcold) < 1400):.0f} %, "
          f"at the floor ({fl:.0f} K) in {100 * np.mean(np.array(tcold) == fl):.0f} %")
    sw = [b for b in m2ref if b in V and V[b]["cls"] == 1]
    if sw:
        t = [parse(V[b]["classes"]["M1"]["best"])["T2"] for b in sw]
        print(f"the {len(sw)} caps that switched M2 -> M1 have best-M1 T2 = {sorted(t)}")
    f21 = [parse(V[b]["classes"]["M2"]["best"]) for b in V if V[b]["cls"] == 2]
    if f21: print(f"f2 > f1 in {100 * np.mean([p['f2'] > p['f1'] for p in f21]):.0f} % of the M2 caps")
