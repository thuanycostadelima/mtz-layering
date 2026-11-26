#!/usr/bin/env python3
"""
Correlation of P(M2) with the TX2011 shear-velocity anomaly at 500-600 km, with uncertainties that respect the overlap
of the caps: (1) a block bootstrap over 20 x 20 degree tiles, resampled with replacement, and (2) random subsets of caps
whose centres are at least 20 degrees apart (no shared records), each giving its own r.  The plain bootstrap over caps is
printed for reference.  Also given: Spearman rank correlation, and the same statistics for the T22 bulk basalt fraction.
    python block_bootstrap.py v7_SS10_fields.csv TX2011_percent.nc [SSall.txt] [--radius 10] [--sep 20] [--n 1000]
"""
import sys, argparse, numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("fields"); ap.add_argument("tomo"); ap.add_argument("t22", nargs="?", default=None)
ap.add_argument("--radius", type=float, default=10.0); ap.add_argument("--sep", type=float, default=20.0); ap.add_argument("--n", type=int, default=1000); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); rng = np.random.default_rng(a.seed)


def gcd(lat1, lon1, lat2, lon2):
    p1, p2, dl = np.radians(lat1), np.radians(lat2), np.radians(lon2 - lon1)
    return np.degrees(np.arccos(np.clip(np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl), -1, 1)))


def capmean(lat, lon, glat, glon, gval, r):
    return np.array([np.average(gval[k], weights=np.cos(np.radians(glat[k]))) if k.any() else np.nan for k in (gcd(la, lo, glat, glon) <= r for la, lo in zip(lat, lon))])


def rank(x): return np.argsort(np.argsort(x)).astype(float)


def stats(x, y, lat, lon, label):
    ok = np.isfinite(x) & np.isfinite(y); x, y, lat, lon = x[ok], y[ok], lat[ok], lon[ok]; n = len(x)
    r0 = np.corrcoef(x, y)[0, 1]; rs = np.corrcoef(rank(x), rank(y))[0, 1]
    # plain bootstrap over caps
    rb = [np.corrcoef(x[i], y[i])[0, 1] for i in (rng.integers(0, n, n) for _ in range(a.n))]
    # block bootstrap over 20 x 20 degree tiles
    tile = (np.floor((lat + 90) / 20) * 100 + np.floor((lon % 360) / 20)).astype(int); tiles = np.unique(tile); nt = len(tiles)
    rblk = []
    for _ in range(a.n):
        pick = tiles[rng.integers(0, nt, nt)]; idx = np.concatenate([np.where(tile == t)[0] for t in pick])
        rblk.append(np.corrcoef(x[idx], y[idx])[0, 1])
    # random non-overlapping subsets (centres >= sep apart)
    D = gcd(lat[:, None], lon[:, None], lat[None, :], lon[None, :]); rsub = []; nsub = []
    for _ in range(a.n):
        order = rng.permutation(n); keep = []
        for i in order:
            if all(D[i, j] >= a.sep for j in keep): keep.append(i)
        keep = np.array(keep); rsub.append(np.corrcoef(x[keep], y[keep])[0, 1]); nsub.append(len(keep))
    pc = lambda v: np.percentile(v, [2.5, 97.5])
    print(f"\n{label}: {n} caps, r = {r0:.2f}, Spearman {rs:.2f}")
    print(f"  bootstrap over caps (treats caps as independent): 95 % interval {pc(rb)[0]:.2f} to {pc(rb)[1]:.2f}")
    print(f"  block bootstrap over {nt} tiles of 20 x 20 degrees:        95 % interval {pc(rblk)[0]:.2f} to {pc(rblk)[1]:.2f}")
    print(f"  {a.n} random subsets of caps >= {a.sep:.0f} degrees apart (median {np.median(nsub):.0f} caps, no shared records): "
          f"median r {np.median(rsub):.2f}, 95 % of subsets {pc(rsub)[0]:.2f} to {pc(rsub)[1]:.2f}, r > 0 in {100 * np.mean(np.array(rsub) > 0):.0f} %")


F = np.genfromtxt(a.fields, delimiter=",", names=True); lat, lon, P = F["lat"], F["lon"], F["P_M2"]
import netCDF4 as nc
d = nc.Dataset(a.tomo); dep = d.variables["depth"][:]; tl = d.variables["latitude"][:]; tn = d.variables["longitude"][:]; dvs = np.array(d.variables["dvs"][:])
L = np.nanmean(dvs[(dep >= 500) & (dep <= 600)], axis=0); LAT, LON = np.meshgrid(tl, tn, indexing="ij")
v = capmean(lat, lon, LAT.ravel(), LON.ravel(), L.ravel(), a.radius)
stats(v, P, lat, lon, "P(M2) vs TX2011 dVs at 500-600 km")
stats(v, F["T_M0"], lat, lon, "uniform-class temperature vs TX2011 dVs")
if a.t22:
    t = np.loadtxt(a.t22); T22 = capmean(lat, lon, t[:, 0], t[:, 1], t[:, 2], a.radius); f22 = capmean(lat, lon, t[:, 0], t[:, 1], t[:, 3], a.radius)
    stats(T22, F["T_M0"], lat, lon, "uniform-class temperature vs T22 temperature")
    stats(f22, P, lat, lon, "P(M2) vs T22 basalt fraction")
