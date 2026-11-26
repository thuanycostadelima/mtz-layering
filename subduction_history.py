#!/usr/bin/env python3
"""
Subduction history beneath each cap from GPlates (Müller et al. 2022 via gplately), for the interpretation figure.

For each cap centre and each reconstruction age (0..200 Ma, step 10), the distance to the nearest subduction-zone
segment at that age is computed. Output per cap:
    age_last   : youngest age at which a trench lay within --reach degrees of the cap centre (NaN if never)
    age_first  : oldest such age within 200 Ma
    n_ages     : number of ages (of 21) with a trench within reach   (~ cumulative slab delivery)
    dmin_0     : present-day distance to the nearest trench (deg)
Then joins with the fields CSV and plots P(M2) and E_df against age_last and n_ages.

Usage: python subduction_history.py v7_SS10_fields.csv figs/figS10_subd.csv figs/figS10_subd [--reach 10]
"""
import argparse, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt


def gcd(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    return np.degrees(np.arccos(np.clip(np.sin(la1) * np.sin(la2) + np.cos(la1) * np.cos(la2) * np.cos(lo1 - lo2), -1, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fields"); ap.add_argument("out_csv"); ap.add_argument("out_fig")
    ap.add_argument("--reach", type=float, default=10.0); ap.add_argument("--step", type=float, default=10.0)
    ap.add_argument("--replot", action="store_true", help="skip the reconstruction and redraw from an existing out_csv")
    a = ap.parse_args()
    d = pd.read_csv(a.fields)
    if a.replot: return plot(pd.read_csv(a.out_csv), a)
    import gplately
    g = gplately.DataServer("Muller2022"); rot, topo, static = g.get_plate_reconstruction_files()
    model = gplately.PlateReconstruction(rot, topo, static)
    ages = np.arange(0, 200 + 1e-9, a.step)
    D = np.full((len(d), len(ages)), np.nan)
    for j, age in enumerate(ages):
        pts = model.tessellate_subduction_zones(age, tessellation_threshold_radians=np.radians(0.5))
        if pts is None or len(pts) == 0:
            continue
        tlon, tlat = pts[:, 0], pts[:, 1]
        for i, (la, lo) in enumerate(zip(d.lat.values, d.lon.values)):
            D[i, j] = gcd(la, lo, tlat, tlon).min()
        print(f"{age:.0f} Ma done", flush=True)
    within = D <= a.reach
    d["dmin_0"] = D[:, 0]
    d["age_last"] = [ages[w][0] if w.any() else np.nan for w in within]
    d["age_first"] = [ages[w][-1] if w.any() else np.nan for w in within]
    d["n_ages"] = within.sum(1)
    d.to_csv(a.out_csv, index=False); plot(d, a)


def plot(d, a):
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "font.family": "sans-serif"})
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9)); fig.subplots_adjust(left=0.08, right=0.985, top=0.9, bottom=0.32, wspace=0.35)
    # (a) P(M2) vs age of last subduction (binned)
    ax = axs[0]; edges = np.r_[-5, 5, 25, 45, 65, 85, 125, 165, 205]; lab = ["0", "10–20", "30–40", "50–60", "70–80", "90–120", "130–160", "170–200", "none"]
    x = d.age_last.fillna(300).values; idx = np.digitize(x, edges) - 1; idx[x > 250] = len(lab) - 1
    means = [d.P_M2.values[idx == k].mean() if (idx == k).any() else np.nan for k in range(len(lab))]
    sem = [d.P_M2.values[idx == k].std() / max(1, np.sqrt((idx == k).sum())) for k in range(len(lab))]
    cnt = [(idx == k).sum() for k in range(len(lab))]
    ax.bar(range(len(lab)), means, yerr=sem, color="C3", alpha=0.7); ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab, rotation=45, ha="right", fontsize=7)
    for k, c in enumerate(cnt): ax.text(k, 0.02, str(c), ha="center", fontsize=7)
    ax.set_xlabel(f"time since a trench lay within {a.reach:g}° (Ma)"); ax.set_ylabel("mean P(M2)"); ax.set_ylim(0, 1); ax.set_title("a   time since subduction", loc="left", fontweight="bold")
    # (b) P(M2) vs cumulative subduction (n_ages)
    ax = axs[1]; nb = np.arange(0, 22, 3)
    idx = np.digitize(d.n_ages.values, nb) - 1
    means = [d.P_M2.values[idx == k].mean() if (idx == k).any() else np.nan for k in range(len(nb) - 1)]
    sem = [d.P_M2.values[idx == k].std() / max(1, np.sqrt((idx == k).sum())) for k in range(len(nb) - 1)]
    ax.bar(range(len(nb) - 1), means, yerr=sem, color="C1", alpha=0.7); ax.set_xticks(range(len(nb) - 1)); ax.set_xticklabels([f"{nb[k]}–{nb[k+1]-1}" for k in range(len(nb) - 1)], fontsize=7, rotation=45, ha="right")
    ax.set_xlabel(f"10-Myr steps with a trench within {a.reach:g}°"); ax.set_ylabel("mean P(M2)"); ax.set_ylim(0, 1); ax.set_title("b   cumulative subduction", loc="left", fontweight="bold")
    # (c) E_df vs n_ages scatter
    ax = axs[2]; db = np.array([0, 5, 10, 15, 20, 30, 40, 60]); idx = np.digitize(d.dmin_0.values, db) - 1
    means = [d.P_M2.values[idx == k].mean() if (idx == k).any() else np.nan for k in range(len(db) - 1)]
    sem = [d.P_M2.values[idx == k].std() / max(1, np.sqrt((idx == k).sum())) for k in range(len(db) - 1)]
    ax.bar(range(len(db) - 1), means, yerr=sem, color="C0", alpha=0.7); ax.set_xticks(range(len(db) - 1)); ax.set_xticklabels([f"{db[k]}–{db[k+1]}" for k in range(len(db) - 1)], fontsize=7, rotation=45, ha="right")
    for k in range(len(db) - 1): ax.text(k, 0.02, str((idx == k).sum()), ha="center", fontsize=6.5)
    ax.set_xlabel("distance to the nearest trench today (°)"); ax.set_ylabel("mean P(M2)"); ax.set_ylim(0, 1); ax.set_title("c   present trench distance", loc="left", fontweight="bold")
    r = np.corrcoef(d.n_ages, d.P_M2)[0, 1]; print(f"caps: {len(d)}; corr(P(M2), n_ages) = {r:.2f}; corr(P(M2), present trench distance) = {np.corrcoef(d.dmin_0, d.P_M2)[0,1]:.2f}")
    for ax in axs: ax.grid(alpha=0.15, lw=0.4)
    fig.savefig(a.out_fig + ".png", dpi=300); fig.savefig(a.out_fig + ".pdf"); print("saved", a.out_fig)


if __name__ == "__main__":
    main()
