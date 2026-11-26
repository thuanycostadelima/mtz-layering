#!/usr/bin/env python3
"""
Write present-day and reconstructed subduction zones (Müller et al. 2022 via gplately) in GMT multi-segment format:
trench_gmt.txt (0 Ma) and trenches_<age>Ma.txt for 20..200 Ma, for the map figures.

    conda activate pygmt; python make_trenches.py
"""
import numpy as np, gplately
g = gplately.DataServer("Muller2022"); rot, topo, static = g.get_plate_reconstruction_files()
model = gplately.PlateReconstruction(rot, topo, static)
for age in range(0, 201, 20):
    pts = model.tessellate_subduction_zones(age, tessellation_threshold_radians=np.radians(0.25)); lon, lat = pts[:, 0], pts[:, 1]
    d = np.r_[0, np.hypot(np.diff(lon), np.diff(lat))]; out = "trench_gmt.txt" if age == 0 else f"trenches_{age}Ma.txt"
    with open(out, "w") as f:
        for x, y, dd in zip(lon, lat, d):
            if dd > 1.0: f.write(">\n")
            f.write(f"{x:.3f} {y:.3f}\n")
    print(f"{out}: {len(lon)} points")
