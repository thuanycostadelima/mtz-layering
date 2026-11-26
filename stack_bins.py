#!/usr/bin/env python3
"""
Synthetic stacks with the real source-receiver geometry of each bin, from the packed
elementary-source library (pack_calib.py) and the per-cap record lists.

For every record in inputs/<phase>/<r>deg_caps/lat_lon.txt:
  1. look up the GCMT moment tensor (ndk catalogue) by the event ID in column 1
  2. take the library depth and distance nearest to the record's
  3. rotate the tensor so that the station lies due south of the source (library geometry)
     and combine the six elementary traces -> transverse (SS) or vertical (PP) trace
  4. band-pass (15-50 s SS / 8-50 s PP), align on the main phase, normalise to its amplitude
Then slant-stack the bin with the SNR weights (column 15) about the reference distance and
sample the vespagram along the (t, slowness) curve of the matching observed cross-section.

Output: <out>/<lat>_<lon>.txt with columns  t  slowness  amp   (same grid as the observed file),
plus <out>/summary.txt (bin, n records used, n missing CMT, ...).

Usage:
  python stack_bins.py inputs/SS/10deg_caps xsec/SS/10deg_caps calib_pack gcmt.ndk out_dir \
         --phase SS --ref 124 [--cols 1,2,3,4,5,6,8,15] [--nproc 4] [--only 0.0_27.0]
--cols: 1-based columns of  eventID, ev lat, ev lon, ev depth, sta lat, sta lon, distance, weight
"""
import argparse, glob, os, re, sys
from functools import lru_cache
import numpy as np
from scipy.signal import butter, sosfiltfilt
from obspy.geodetics import gps2dist_azimuth
from obspy.taup import TauPyModel

TENSORS = ("Mrr", "Mtt", "Mpp", "Mrt", "Mrp", "Mtp")
# (row, col) in the 3x3 symmetric tensor in the (r, theta, phi) frame
IDX = {"Mrr": (0, 0), "Mtt": (1, 1), "Mpp": (2, 2), "Mrt": (0, 1), "Mrp": (0, 2), "Mtp": (1, 2)}


# ---------------------------------------------------------------- GCMT catalogue
def read_ndk(paths):
    """Return dict id -> (mrr, mtt, mpp, mrt, mrp, mtp) in N m, plus date-time index."""
    cat, byt = {}, {}
    for p in paths:
        lines = open(p, errors="replace").read().splitlines()
        for k in range(0, len(lines) - 4, 5):
            l1, l2, l4 = lines[k], lines[k + 1], lines[k + 3]
            eid = l2[:16].strip()
            try:
                ex = int(l4[:2]); v = l4[2:].split()
                mt = np.array([float(v[i]) for i in (0, 2, 4, 6, 8, 10)]) * 10.0 ** ex * 1e-7  # dyne cm -> N m
            except Exception:
                continue
            cat[eid] = mt
            cat[eid.lstrip("CMBSGHZ")] = mt
            # record IDs are YYYYMMDD + letter: index that too
            core = eid.lstrip("CMBSGHZ")
            if len(core) == 7 and core[:6].isdigit():          # old style MMDDYY + letter
                mm, dd, yy = core[:2], core[2:4], core[4:6]
                yyyy = ("19" if int(yy) >= 76 else "20") + yy
                cat[yyyy + mm + dd + core[6:]] = mt
            elif len(core) >= 13 and core[:12].isdigit():      # YYYYMMDDHHMM + letter
                cat[core[:8] + core[12:]] = mt
            # origin time from line 1: "PDEW2004/12/26 00:58:50.0 ..."
            m = re.search(r"(\d{4})/(\d{2})/(\d{2}) (\d{2}):(\d{2}):(\d{2})", l1)
            if m:
                byt["".join(m.groups()[:5])] = mt
    return cat, byt


def lookup(cat, byt, eid):
    s = eid.strip()
    for key in (s, s.lstrip("CMBSGHZ"), s.upper()):
        if key in cat:
            return cat[key]
    m = re.search(r"(\d{12})", s)
    if m and m.group(1) in byt:
        return byt[m.group(1)]
    return None


# ---------------------------------------------------------------- rotation
def rotate_to_library(mt6, az):
    """Moment tensor (r,theta,phi) at the real source -> frame in which the station is due south.
    e1 = r, e2 = horizontal unit vector towards the station, e3 = r x e2 (transverse)."""
    M = np.zeros((3, 3))
    for name, (i, j) in IDX.items():
        M[i, j] = M[j, i] = mt6[TENSORS.index(name)]
    a = np.radians(az)
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.array([0.0, -np.cos(a), np.sin(a)])      # theta = south, phi = east
    e3 = np.cross(e1, e2)
    E = np.vstack([e1, e2, e3])
    Mp = E @ M @ E.T
    return np.array([Mp[IDX[n]] for n in TENSORS])


# ---------------------------------------------------------------- library
class Library:
    def __init__(self, pack_dir):
        self.dir = pack_dir
        files = glob.glob(os.path.join(pack_dir, "*_z*.npz"))
        self.depths = sorted({float(re.search(r"_z([\d.]+)\.npz", f).group(1)) for f in files})
        self.tensors = sorted({os.path.basename(f).split("_z")[0] for f in files})
        z = np.load(files[0]); self.t = z["t"].astype(float); self.dist = z["dist"].astype(float)
        self.dt = float(self.t[1] - self.t[0])

    @lru_cache(maxsize=None)
    def load(self, tensor, depth, comp):
        # one component only: 6 tensors x 30 depths x 181 x 3601 float32 = ~470 MB resident
        return np.load(os.path.join(self.dir, f"{tensor}_z{depth:05.1f}.npz"))[comp]

    def trace(self, mt6_rot, depth, dist, comp):
        zi = self.depths[int(np.argmin(np.abs(np.array(self.depths) - depth)))]
        ri = int(np.argmin(np.abs(self.dist - dist)))
        out = np.zeros(len(self.t))
        for k, name in enumerate(TENSORS):
            if name not in self.tensors or mt6_rot[k] == 0.0:
                continue
            out += mt6_rot[k] / 1e26 * self.load(name, zi, comp)[ri]
        return out


# ---------------------------------------------------------------- processing
def bandpass(x, dt, tmin, tmax):
    sos = butter(4, [1.0 / tmax, 1.0 / tmin], btype="band", fs=1.0 / dt, output="sos")
    return sosfiltfilt(sos, x)


_taup = TauPyModel("prem")
@lru_cache(maxsize=4096)
def main_time(phase, depth, dist):
    arr = _taup.get_travel_times(source_depth_in_km=depth, distance_in_degree=dist, phase_list=[phase])
    return arr[0].time if arr else np.nan


def process_record(lib, mt6, az, depth, dist, phase):
    comp = "phi" if phase == "SS" else "r"
    x = lib.trace(rotate_to_library(mt6, az), depth, dist, comp)
    x = bandpass(x, lib.dt, 15.0 if phase == "SS" else 8.0, 50.0)
    t0 = main_time(phase, round(depth, 1), round(dist, 2))
    if not np.isfinite(t0):
        return None
    w = (lib.t > t0 - 60) & (lib.t < t0 + 60)
    if not w.any():
        return None
    k = np.argmax(np.abs(x[w])); tpk = lib.t[w][k]; apk = x[w][k]
    return (lib.t - tpk), x / apk          # time relative to main-phase peak, normalised, main phase positive


def slant_stack(traces, dists, weights, ref, tgrid, pcurve):
    """traces: list of (trel, x); sample the weighted slant stack along the (t, p) curve.
    For each output time t with slowness p (s/deg, relative to the main phase), the stack is
    sum_i w_i x_i(t + p (D_i - ref)) / sum w_i."""
    out = np.zeros(len(tgrid)); wsum = 0.0
    for (trel, x), D, w in zip(traces, dists, weights):
        ts = tgrid + pcurve * (D - ref)
        out += w * np.interp(ts, trel, x, left=0.0, right=0.0); wsum += w
    return out / wsum if wsum > 0 else out * np.nan


# ---------------------------------------------------------------- main
def one_bin(args):
    inp, obs, lib, cat, byt, phase, ref, cols, out, fixed = args
    name = os.path.basename(inp)
    try:
        od = np.loadtxt(obs)
    except Exception:
        return name, "no observed cross-section"
    tgrid, pcurve = od[:, 0], od[:, 1]
    traces, dists, weights = [], [], []
    n_miss = n_bad = 0
    for line in open(inp):
        f = line.split()
        if len(f) < max(cols):
            continue
        try:
            eid = f[cols[0] - 1].split(".")[0]; evla, evlo, evdp = (float(f[c - 1]) for c in cols[1:4])
            stla, stlo = (float(f[c - 1]) for c in cols[4:6]); D = float(f[cols[6] - 1]); w = float(f[cols[7] - 1])
        except ValueError:
            n_bad += 1; continue
        mt = lookup(cat, byt, eid)
        if mt is None:
            n_miss += 1; continue
        _, az, _ = gps2dist_azimuth(evla, evlo, stla, stlo)
        if fixed:   # ignore mechanism and depth: M_rphi (SS) / explosion (PP) at 10 km in the library frame
            mt = np.array([0, 0, 0, 0, 1e26, 0.0]) if phase == "SS" else np.array([1e26, 1e26, 1e26, 0, 0, 0.0])
            az, evdp = 180.0, 10.0
        r = process_record(lib, mt, az, max(evdp, 0.0), D, phase)
        if r is None:
            n_bad += 1; continue
        traces.append(r); dists.append(D); weights.append(max(w, 0.0))
    if len(traces) < 1:
        return name, f"only {len(traces)} usable records ({n_miss} no CMT, {n_bad} bad)"
    st = slant_stack(traces, np.array(dists), np.array(weights), ref, tgrid, pcurve)
    np.savetxt(out, np.c_[tgrid, pcurve, st], fmt="%.2f %.5f %.6e")
    return name, f"ok n={len(traces)} miss={n_miss} bad={n_bad}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs_dir"); ap.add_argument("xsec_dir"); ap.add_argument("pack_dir")
    ap.add_argument("ndk", help="ndk file(s), comma-separated"); ap.add_argument("out_dir")
    ap.add_argument("--phase", default="SS"); ap.add_argument("--ref", type=float, default=124.0)
    ap.add_argument("--cols", default="1,2,3,4,5,6,8,15")
    ap.add_argument("--nproc", type=int, default=1); ap.add_argument("--only", default=None)
    ap.add_argument("--fixed-source", action="store_true",
                    help="use a fixed source (M_rphi for SS, explosion for PP, 10 km) for every record: isolates the distance effect")
    a = ap.parse_args()
    cols = [int(c) for c in a.cols.split(",")]
    cat, byt = read_ndk(a.ndk.split(","))
    print(f"GCMT: {len(byt)} events", flush=True)
    lib = Library(a.pack_dir)
    print(f"library: tensors {lib.tensors}, {len(lib.depths)} depths, {len(lib.dist)} distances {lib.dist[0]}-{lib.dist[-1]}", flush=True)
    os.makedirs(a.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.inputs_dir, "*.txt")))
    if a.only:
        files = [f for f in files if os.path.basename(f).startswith(a.only)]
    jobs = []
    for f in files:
        name = os.path.basename(f)
        obs = os.path.join(a.xsec_dir, name)
        out = os.path.join(a.out_dir, name)
        if os.path.exists(out):
            continue
        jobs.append((f, obs, lib, cat, byt, a.phase, a.ref, cols, out, a.fixed_source))
    log = open(os.path.join(a.out_dir, "summary.txt"), "a")
    if a.nproc > 1:
        from multiprocessing import Pool
        it = Pool(a.nproc).imap_unordered(one_bin, jobs)
    else:
        it = map(one_bin, jobs)
    for name, status in it:
        print(name, status, file=log, flush=True); print(name, status, flush=True)


if __name__ == "__main__":
    main()
