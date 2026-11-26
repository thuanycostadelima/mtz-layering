#!/usr/bin/env python3
"""
Convert a TauP-style .tvel profile (depth km, Vp, Vs km/s, rho g/cc; header 2 lines)
into a Yspec model deck, taking Q_kappa and Q_mu from a reference Yspec deck (prem.model).

Usage
-----
  python tvel2yspec.py prem.model input.tvel output.model
  python tvel2yspec.py prem.model --batch IN_DIR OUT_DIR      # every .tvel under IN_DIR

Deck layout (as in prem.model):
  line 1  : " AXISEM model for YSPEC: "
  line 2  : flags line, copied from the reference deck
  line 3  : N  N_ic  N_oc   (total rows, last inner-core row, last outer-core row; 1-based)
  rows    : radius(m)  rho(kg/m3)  Vp(m/s)  Vs(m/s)  Qkappa  Qmu, centre outward,
            discontinuities as duplicated radii.
Fluid outer core is written with Vs = 0 exactly (the 0.001 placeholder in some
.tvel files is removed).
"""
import argparse
import os
import sys
import numpy as np

R_EARTH_KM = 6371.0


def read_ref_deck(path):
    with open(path) as fh:
        lines = fh.read().splitlines()
    flags = lines[1]
    rows = np.loadtxt(lines[3:])
    return flags, rows  # radius m, rho, vp, vs, Qk, Qmu


def read_tvel(path):
    with open(path) as fh:
        lines = fh.read().splitlines()
    return np.loadtxt(lines[2:])  # depth km, vp, vs, rho


def q_at_radius(ref, r):
    """Q_kappa, Q_mu of the reference layer containing radius r (m).
    For a radius on a reference discontinuity, the layer *below* is used for the
    row written first (deeper side) and the layer above for the second: handled
    by the caller passing side=-1/+1."""
    rr = ref[:, 0]
    # last reference row whose radius <= r; duplicated radii mean we land on the
    # upper side of a reference discontinuity, which is what we want for a row
    # strictly inside a layer.
    idx = np.searchsorted(rr, r, side="right") - 1
    idx = max(0, min(idx, len(rr) - 1))
    return ref[idx, 4], ref[idx, 5]


def convert(ref_flags, ref, tvel):
    depth = tvel[:, 0]
    r = (R_EARTH_KM - depth) * 1e3          # m, decreasing with row index
    rho = tvel[:, 3] * 1e3
    vp = tvel[:, 1] * 1e3
    vs = tvel[:, 2] * 1e3
    vs = np.where(vs < 5.0, 0.0, vs)        # fluid placeholder -> exactly 0

    # reorder centre outward: reverse first so that at a duplicated radius the
    # deeper-side row (listed second in the depth-ordered .tvel) comes first
    r, rho, vp, vs = r[::-1], rho[::-1], vp[::-1], vs[::-1]
    order = np.argsort(r, kind="stable")
    r, rho, vp, vs = r[order], rho[order], vp[order], vs[order]

    # Q from reference deck. At a duplicated radius the first (deeper) row should
    # take Q from just below, the second from just above; nudge by 1 m.
    qk = np.empty_like(r)
    qm = np.empty_like(r)
    n = len(r)
    for i in range(n):
        if i + 1 < n and r[i + 1] == r[i]:
            probe = r[i] - 1.0
        elif i > 0 and r[i - 1] == r[i]:
            probe = r[i] + 1.0
        else:
            probe = r[i]
        qk[i], qm[i] = q_at_radius(ref, probe)
    qm = np.where(vs == 0.0, 0.0, qm)

    # layer indices: last row with vs==0 run from the centre side
    fluid = vs == 0.0
    oc_rows = np.where(fluid)[0]
    if len(oc_rows) == 0:
        raise ValueError("no fluid outer core found (Vs==0 rows)")
    n_ic = int(oc_rows[0])            # 1-based index of last inner-core row == 0-based first fluid row
    n_oc = int(oc_rows[-1]) + 1       # 1-based index of last fluid row
    deck = np.column_stack([r, rho, vp, vs, qk, qm])
    return ref_flags, n, n_ic, n_oc, deck


def write_deck(path, flags, n, n_ic, n_oc, deck, label):
    with open(path, "w") as fh:
        fh.write(f" AXISEM model for YSPEC: {label}\n")
        fh.write(f"{flags}\n")
        fh.write(f"{n:12d}{n_ic:12d}{n_oc:12d}\n")
        for row in deck:
            fh.write(f"{row[0]:9.0f}. {row[1]:8.2f} {row[2]:8.2f} {row[3]:8.2f} {row[4]:8.1f} {row[5]:8.1f}\n")


def check_against_ref(ref):
    """Self-test: line-3 counts recomputed from the reference deck must match its header."""
    vs = ref[:, 3]
    fluid = np.where(vs == 0.0)[0]
    return int(fluid[0]), int(fluid[-1]) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("inp", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--batch", nargs=2, metavar=("IN_DIR", "OUT_DIR"))
    a = ap.parse_args()

    flags, ref = read_ref_deck(a.ref)
    with open(a.ref) as fh:
        hdr3 = fh.read().splitlines()[2].split()
    n_ic, n_oc = check_against_ref(ref)
    if (int(hdr3[1]), int(hdr3[2])) != (n_ic, n_oc):
        sys.exit(f"reference deck header {hdr3} does not match recomputed counts {n_ic},{n_oc}")

    jobs = []
    if a.batch:
        in_dir, out_dir = a.batch
        for root, _, files in os.walk(in_dir):
            for f in sorted(files):
                if f.endswith(".tvel"):
                    rel = os.path.relpath(os.path.join(root, f), in_dir)
                    jobs.append((os.path.join(root, f),
                                 os.path.join(out_dir, rel[:-5] + ".model")))
    else:
        if not (a.inp and a.out):
            ap.error("give input.tvel output.model, or --batch IN_DIR OUT_DIR")
        jobs.append((a.inp, a.out))

    for src, dst in jobs:
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        label = os.path.basename(src)[:-5]
        _, n, nic, noc, deck = convert(flags, ref, read_tvel(src))
        write_deck(dst, flags, n, nic, noc, deck, label)
    print(f"converted {len(jobs)} model(s); last: N={n} N_ic={nic} N_oc={noc}")


if __name__ == "__main__":
    main()
