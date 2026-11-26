#!/usr/bin/env python3
"""
Supplementary figures S4 (pick-sensitivity table), S5, S6, S7, S8, S9 and S11, generated from the files in mtzsyn/.

    python suppfigs.py all             # all of them
    python suppfigs.py figS6 figS7     # selected ones
    python suppfigs.py list

S1 is made by fig1.py --suppl, S2-S4 by figS.py, S10 by subduction_history.py and S12 by fig4.py --m1.
All paths are in CFG below. Output: figs/<name>.png and .pdf.
"""
import os, sys, glob, json, re
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7, "font.family": "sans-serif"})

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = HERE if os.path.exists(os.path.join(HERE, "prem_sds_S.npz")) else os.path.dirname(HERE)   # works from mtzsyn/ or mtzsyn/code/
sys.path.insert(0, HERE); sys.path.insert(0, ROOT); os.chdir(ROOT)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "legend.fontsize": 8})

CFG = dict(
    obs="crosssections_2024_SNR/SS/10deg_caps", inputs="crosssections_2024_SNR/SSinputs/10deg_caps",
    run="inv_v7_SS10",                                   # reference inversion (invert_bins2: 50 K library, correlated noise, marginal A/tau, noise-fit)
    results="v7_SS10_results.csv", fields="v7_SS10_fields.csv",   # from export_fields.py
    lib=["lib_shift_0.npz", "lib_shift_1.npz", "lib_shift_2.npz", "lib_shift_3.npz", "lib_shift50_0.npz"],
    mech="mech_filters_SS10_v7.npz", mech_B="mech_filters_SS10_B_v7.npz",
    stacks_real="calib_stacks_v7/SS10", stacks_fixed="calib_stacks_v7/SS10_fixed",
    stacks_B_real="calib_stacks_v7/SS10_B", stacks_B_fixed="calib_stacks_v7/SS10_B_fixed",
    pp_real="calib_stacks_v7/PP10", pp_fixed="calib_stacks_v7/PP10_fixed", mech_pp="mech_filters_PP10_v7.npz",
    table_S="prem_sds_S.npz", table_P="prem_sds_P.npz",
    models="models",
    picks_ss="picks_SS_10_v7.csv", picks_pp="picks_PP105_10_v7.csv",
    picks_ss_r={5: "picks_SS_5_v7.csv", 7.5: "picks_SS_7.5_v7.csv", 10: "picks_SS_10_v7.csv", 15: "picks_SS_15_v7.csv"},
    picks_pp_r={5: "picks_PP105_5_v7.csv", 7.5: "picks_PP105_7.5_v7.csv", 10: "picks_PP105_10_v7.csv", 15: "picks_PP105_15_v7.csv"},
    recovery={"M0: f=0.2, 1600 K": "rec_v7_MM0.2_1600", "M1: ΔT=−200 K": "rec_v7_MM0.2-0.2_1700-1500_zc523_w0",
              "M2: Δf=+0.2, ΔT=+100 K": "rec_v7_MM0.2-0.4_1600-1700_zc523_w0", "M2: Δf=+0.4, ΔT=0": "rec_v7_MM0.2-0.6_1700-1700_zc523_w0"},
    sensitivity="sensitivity_v7.csv",                  # table of variants (made on the reference machine with compare_runs.py; shipped with the package)
    out="figs",
)
TG = np.arange(-340.0, -70.0 + 1e-9, 1.0); WIN = (TG >= -300) & (TG <= -110)


def out(name, fig):
    os.makedirs(CFG["out"], exist_ok=True); fig.savefig(f"{CFG['out']}/{name}.png", dpi=200, bbox_inches="tight"); fig.savefig(f"{CFG['out']}/{name}.pdf", bbox_inches="tight"); plt.close(fig); print("saved", name)


def depth_fn(table, dist):
    tab = np.load(table); j = np.argmin(np.abs(tab["dist"] - dist)); dt = tab["dt"][:, j]; ok = np.isfinite(dt)
    return (lambda t: np.interp(t, dt[ok][::-1], tab["depth"][ok][::-1])), (lambda z: np.interp(z, tab["depth"][ok], dt[ok]))


def pk(x, t, lo, hi, sign=1):
    w = (t > lo) & (t < hi); k = np.argmax(sign * x[w]); return x[w][k], t[w][k]


def gcd(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    return np.degrees(np.arccos(np.clip(np.sin(la1) * np.sin(la2) + np.cos(la1) * np.cos(la2) * np.cos(lo1 - lo2), -1, 1)))


def load_run(run):
    """P(M0,M1,M2), class, posterior means per cap from an invert_bins2 directory."""
    R = {}
    for f in sorted(glob.glob(f"{run}/*.json")):
        J = json.load(open(f)); c = J["classes"]; lz = np.array([c[k]["logZ"] for k in ("M0", "M1", "M2")]); p = np.exp(lz - lz.max()); p /= p.sum()
        R[J["bin"]] = dict(P=p, cls=2 if J["lnB21"] > 3 else (1 if J["lnB10"] > 3 else 0), lat=J["lat"], lon=J["lon"], chi2=J["chi2_red_best"], J=J)
    return R


def classes_from_summary(path):
    r = np.genfromtxt(path, delimiter=",", names=True); g = r["chi2"] < 2; m2 = (r["lnB21"] > 3) & g; m1 = (r["lnB10"] > 3) & ~m2 & g
    return r, g, np.where(~g, -1, np.where(m2, 2, np.where(m1, 1, 0)))


# ----------------------------------------------------------------------------- Supplementary
def figS4_picks():
    os.system(f"{sys.executable} {os.path.join(ROOT, 'pick_sensitivity.py')} {CFG['obs']} {CFG['out']}/figS4_picks.csv {CFG['out']}/figS4_picks.png --table {CFG['table_S']}")


def figS5():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ip", os.path.join(ROOT, "interp_profiles.py"))
    src = open(spec.origin).read().split("os.makedirs")[0].replace("MD, OUT = sys.argv[1], sys.argv[2]", f"MD, OUT = '{CFG['models']}', 'models50'")
    ns = {}; exec(src, ns); tvel, interp = ns["tvel"], ns["interp"]
    fig = plt.figure(figsize=(7.2, 6.0)); gs = plt.GridSpec(2, 2, figure=fig, left=0.09, right=0.985, top=0.95, bottom=0.1, hspace=0.45, wspace=0.3); axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
    A, _ = tvel(0.2, 1500); B, _ = tvel(0.2, 1700); C, _ = tvel(0.2, 1600); P = interp(A, B); z = C[:, 0]; m = (z > 330) & (z < 760)
    ax = axs[0]; ax.plot(A[m, 2], A[m, 0], "C0", lw=0.9, label="1500 K"); ax.plot(B[m, 2], B[m, 0], "C3", lw=0.9, label="1700 K"); ax.plot(C[m, 2], C[m, 0], "k", lw=2, alpha=.4, label="1600 K (Perple_X)"); ax.plot(P[m, 2], P[m, 0], "k--", lw=1, label="1600 K, interpolated\nfrom 1500 and 1700 K")
    ax.set_ylim(760, 330); ax.set_xlabel("V$_s$ (km s$^{-1}$)"); ax.set_ylabel("depth (km)"); ax.legend(frameon=False, loc="lower left"); ax.set_title("a   phase-aware interpolation, f = 0.2", loc="left", fontweight="bold")
    ax = axs[1]; ax.plot((P[m, 2] - C[m, 2]) * 1000, P[m, 0], "k", label="phase-aware"); naive = 0.5 * (A[:, 2] + B[:, 2]); ax.plot((naive[m] - C[m, 2]) * 1000, C[m, 0], "C1", lw=0.9, label="naive (same depth)"); ax.set_ylim(760, 330); ax.set_xlabel("V$_s$ error (m s$^{-1}$)"); ax.axvline(0, color="0.6", lw=0.5); ax.legend(frameon=False); ax.set_title("b   error against the computed 1600 K profile", loc="left", fontweight="bold"); ax.tick_params(labelleft=False)
    ax = axs[2]; rows = []
    for f in (0.0, 0.2, 0.4, 0.6):
        for T in (1500, 1600, 1700, 1800):
            A, _ = tvel(f, T - 100); B, _ = tvel(f, T + 100); C, _ = tvel(f, T); P = interp(A, B); z = C[:, 0]; m = (z > 300) & (z < 800)
            rows.append((f, T, np.sqrt(np.mean((P[m, 2] - C[m, 2]) ** 2)) * 1000, np.sqrt(np.mean((0.5 * (A[m, 2] + B[m, 2]) - C[m, 2]) ** 2)) * 1000, np.sqrt(np.mean((A[m, 2] - C[m, 2]) ** 2)) * 1000))
    R = np.array(rows); x = np.arange(len(R)); ax.bar(x - 0.25, R[:, 2], 0.25, label="phase-aware interpolation"); ax.bar(x, R[:, 3], 0.25, label="naive interpolation"); ax.bar(x + 0.25, R[:, 4], 0.25, label="100 K neighbour (no interpolation)")
    ax.set_xticks(x); ax.set_xticklabels([f"f = {r[0]:.1f}\n{int(r[1])} K" for r in R], fontsize=6.5); ax.set_ylabel("RMS V$_s$ error, 300–800 km (m s$^{-1}$)"); ax.legend(frameon=False, ncol=3, loc="upper left"); ax.set_title("c   200 K-span test at every 100 K node", loc="left", fontweight="bold"); ax.grid(alpha=0.15, lw=0.4, axis="y"); ax.set_ylim(0, R[:, 2:].max() * 1.25)
    out("figS5_interp", fig)


def figS6():
    from invert_bins2 import load_lib, bin_weights, TW, DT
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw=dict(width_ratios=[0.8, 1.1, 1.1])); fig.subplots_adjust(left=0.12, right=0.985, top=0.9, bottom=0.2, wspace=0.4)
    # (a) recovery confusion matrix
    C, labs = [], []
    for lab, d in CFG["recovery"].items():
        if not os.path.exists(f"{d}/summary.csv"): continue
        r, g, cls = classes_from_summary(f"{d}/summary.csv"); C.append([np.mean(cls[g] == k) for k in (0, 1, 2)]); labs.append(lab)
    C = np.array(C); ax = axs[0]; ax.imshow(C, cmap="Blues", vmin=0, vmax=1); ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["M0", "M1", "M2"]); ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=8)
    for i in range(len(labs)):
        for j in range(3): ax.text(j, i, f"{C[i, j] * 100:.0f} %", ha="center", va="center", color="w" if C[i, j] > 0.5 else "k", fontsize=7.5)
    ax.set_xlabel("assigned class"); ax.set_ylabel("true model"); ax.set_title("a   recovery tests", loc="left", fontweight="bold"); ax.tick_params(length=0)
    # (b) model independence of the mechanism filter
    A = np.load(CFG["mech"]); B = np.load(CFG["mech_B"]); L = int(A["L"]); HA = dict(zip(A["names"], A["h"])); HB = dict(zip(B["names"], B["h"])); res = []
    for f in sorted(glob.glob(f"{CFG['stacks_B_fixed']}/*.txt")):
        nm = os.path.basename(f)[:-4]
        if nm == "summary" or nm not in HA: continue
        fx = np.loadtxt(f); fx = np.interp(TG, fx[:, 0], fx[:, 2]); rc = np.loadtxt(f"{CFG['stacks_B_real']}/{nm}.txt"); rc = np.interp(TG, rc[:, 0], rc[:, 2])
        X = np.array([np.roll(fx, k) for k in range(-L, L + 1)]).T; v = lambda y: np.sum((y[WIN] - rc[WIN]) ** 2) / np.sum(rc[WIN] ** 2); res.append((v(fx), v(X @ HA[nm]), v(X @ HB[nm])))
    R = np.array(res) * 100; ax = axs[1]; b = np.linspace(0, 20, 41)
    ax.hist(R[:, 0], bins=b, alpha=.6, label=f"no correction ({np.median(R[:, 0]):.1f} %)"); ax.hist(R[:, 1], bins=b, alpha=.6, label=f"filter from model A ({np.median(R[:, 1]):.1f} %)"); ax.hist(R[:, 2], bins=b, alpha=.6, label=f"own filter ({np.median(R[:, 2]):.1f} %)")
    ax.set_xlabel("residual variance, model B (% of signal)"); ax.set_ylabel("caps"); ax.legend(frameon=False, title="median", title_fontsize=7, loc="center right"); ax.set_title("b   filter independence", loc="left", fontweight="bold"); ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
    # (c) timing residuals
    names, S, meta, tg, dist = load_lib(CFG["lib"]); M = dict(zip(names, range(len(names)))); mech = dict(zip(*[np.load(CFG["mech"])[k] for k in ("names", "h")]))
    tobs = np.arange(TW[0], TW[1] + 1e-9, DT); i0 = int(round(TW[0] - tg[0])); nt = tobs.size; rows = []
    for name, v in load_run(CFG["run"]).items():
        best = max(v["J"]["classes"].values(), key=lambda r: r["best_ll"]); W, _ = bin_weights(f"{CFG['inputs']}/{name}.txt", dist)
        g = np.convolve(W @ S[M[best["best"]]], mech[name], mode="same"); tau = int(best["best_tau"]); g = g[i0 + tau:i0 + tau + nt]
        t, p, x, sg = np.loadtxt(f"{CFG['obs']}/{name}.txt", unpack=True); d = np.interp(tobs, t, x)
        _, t4 = pk(g, tobs, -185, -130); _, t6 = pk(g, tobs, -260, -195); rows.append((pk(d, tobs, t4 - 15, t4 + 15)[1] - t4, pk(d, tobs, t6 - 15, t6 + 15)[1] - t6, tau))
    R = np.array(rows); ax = axs[2]; b = np.arange(-12.5, 13, 1)
    ax.hist(R[:, 2], bins=b, alpha=.6, label=f"shift τ ({np.median(R[:, 2]):+.0f} s)"); ax.hist(R[:, 0], bins=b, alpha=.6, label=f"410 residual ({np.median(R[:, 0]):+.1f} s)"); ax.hist(R[:, 1], bins=b, alpha=.6, label=f"660 residual ({np.median(R[:, 1]):+.1f} s)")
    ax.set_xlabel("data − best-fitting synthetic (s)"); ax.set_ylabel("caps"); ax.legend(frameon=False, title="median", title_fontsize=7, loc="upper right"); ax.set_title("c   timing after the fit", loc="left", fontweight="bold"); ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
    out("figS6_tests", fig)


def figS7():
    """Sensitivity table as a figure, from sensitivity_v7.csv (variant, caps, M0, M1, M2 at B>20, M0/M1/M2 at B>10, f2>f1, r_PM2, same_class)."""
    import csv
    rows = list(csv.reader(open(CFG["sensitivity"]))); head, body = rows[0], rows[1:]
    fig, ax = plt.subplots(figsize=(7.2, 0.3 * len(body) + 0.6)); ax.axis("off"); fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    tb = ax.table(cellText=body, colLabels=head, loc="center", cellLoc="center"); tb.auto_set_font_size(False); tb.set_fontsize(7); tb.scale(1, 1.3); tb.auto_set_column_width(list(range(len(head))))
    for (i, j), c in tb.get_celld().items():
        if i == 0: c.set_text_props(fontweight="bold")
        if i == 1: c.set_facecolor("#fdf0ec")
    out("figS7_sensitivity", fig)


def figS8():
    import pandas as pd
    r = pd.read_csv(CFG["fields"]); lat, lon = np.radians(r.lat.values), np.radians(r.lon.values)
    cosd = np.sin(lat)[:, None] * np.sin(lat)[None, :] + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.cos(lon[:, None] - lon[None, :]); d = np.degrees(np.arccos(np.clip(cosd, -1, 1))); nb = (d > 0.1) & (d <= 15)
    P = r.P_M2.values; nbm = np.array([P[nb[i]].mean() for i in range(len(P))]); cls = np.where(P > 0.9, 2, np.where(P < 0.1, 0, 1))
    same = np.array([(cls[np.where(nb[i])[0]] == cls[i]).mean() for i in range(len(P))]); rng = np.random.default_rng(0)
    exp = [np.mean([(c[np.where(nb[i])[0]] == c[i]).mean() for i in range(len(P))]) for c in (rng.permutation(cls) for _ in range(300))]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0)); fig.subplots_adjust(left=0.09, right=0.985, top=0.9, bottom=0.17, wspace=0.3)
    ax = axs[0]; ax.scatter(nbm, P, s=8, c="C3", alpha=.6, lw=0); ax.set_xlabel("mean P(M2) of neighbouring caps (centres within 15°)"); ax.set_ylabel("P(M2) of the cap"); ax.set_title(f"a   spatial coherence, r = {np.corrcoef(nbm, P)[0, 1]:.2f}", loc="left", fontweight="bold"); ax.grid(alpha=0.15, lw=0.4)
    ax = axs[1]; ax.hist(exp, bins=30, color="0.7", label="random reassignment"); ax.axvline(same.mean(), color="C3", lw=2, label=f"observed ({same.mean():.2f})"); ax.set_xlabel("fraction of neighbours in the same class"); ax.set_ylabel("random trials"); ax.legend(frameon=False); ax.set_title("b   class agreement between neighbours", loc="left", fontweight="bold")
    out("figS8_coherence", fig)


def figS9():
    import pandas as pd
    ss = pd.read_csv(CFG["picks_ss"]); pp = pd.read_csv(CFG["picks_pp"]); mm = ss.merge(pp, on=["lat", "lon"], suffixes=("_ss", "_pp"))
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7)); fig.subplots_adjust(left=0.08, right=0.985, top=0.9, bottom=0.2, wspace=0.4)
    ax = axs[0]; g = (mm.sig410_ss > 5) & (mm.sig410_pp > 5); ax.scatter(mm.z410_ss, mm.z410_pp, s=6, c="0.7", lw=0, label="all caps"); ax.scatter(mm.z410_ss[g], mm.z410_pp[g], s=8, c="C3", lw=0, label=f"410 > 5σ in both\n(r = {np.corrcoef(mm.z410_ss[g], mm.z410_pp[g])[0, 1]:.2f})"); ax.plot([385, 455], [385, 455], "k:", lw=0.8); ax.set_xlabel("SS 410 depth (km)"); ax.set_ylabel("PP 410 depth (km)"); ax.legend(frameon=False, loc="upper left"); ax.set_title(f"a   410 depth, {len(mm)} caps", loc="left", fontweight="bold")
    ax = axs[1]; ax.hist(mm.z660_ss, bins=np.arange(600, 780, 6), alpha=.6, label=f"SS (median {mm.z660_ss.median():.0f})"); ax.hist(mm.z660_pp, bins=np.arange(600, 780, 6), alpha=.6, label=f"PP (median {mm.z660_pp.median():.0f})"); ax.set_xlabel("apparent 660 depth (km)"); ax.set_ylabel("caps"); ax.legend(frameon=False); ax.set_title("b   660 depth", loc="left", fontweight="bold")
    def det(path):
        p = pd.read_csv(path); zpos = np.full(len(p), np.nan)
        for k in ("1", "2", "3"):
            sel = np.isnan(zpos) & (p["pol" + k] > 0); zpos[sel] = p["z" + k][sel]
        return np.isfinite(zpos).mean() * 100
    rs = sorted(CFG["picks_pp_r"]); ax = axs[2]
    ax.plot(rs, [det(CFG["picks_ss_r"][r]) for r in rs], "o-", label="SS"); ax.plot(rs, [det(CFG["picks_pp_r"][r]) for r in rs], "s-", label="PP"); ax.set_xlabel("cap radius (°)"); ax.set_ylabel("caps with a mid-MTZ pick (%)"); ax.legend(frameon=False); ax.set_ylim(0, 100); ax.set_title("c   pick rate", loc="left", fontweight="bold"); ax.grid(alpha=0.15, lw=0.4)
    out("figS9_PP", fig)


def figS11():
    os.system(f"{sys.executable} {os.path.join(ROOT, 'plot_all_fits.py')} {CFG['run']} {CFG['obs']} {CFG['inputs']} {CFG['out']}/figS11_all_fits.pdf --lib {' '.join(CFG['lib'])} --mech {CFG['mech']} --table {CFG['table_S']}")


FIGS = {"figS4_picks": figS4_picks, "figS5": figS5, "figS6": figS6, "figS7": figS7, "figS8": figS8, "figS9": figS9, "figS11": figS11}   # keys = figure numbers in the paper

if __name__ == "__main__":
    args = sys.argv[1:] or ["list"]
    if args == ["list"]:
        print("supplementary figures made here:", " ".join(FIGS)); print("main figures: python fig1.py ... fig6.py (fig1, fig4, fig5 in the pygmt env); S1: fig1.py --suppl; S2-S4: figS.py; S10: subduction_history.py; S12: fig4.py --m1"); sys.exit()
    for k in (FIGS if args == ["all"] else args):
        print("==", k); FIGS[k]()
