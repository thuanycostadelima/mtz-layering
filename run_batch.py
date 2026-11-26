#!/usr/bin/env python3
"""
Run Yspec + post-processing for every .model in a directory, N models in parallel.

Layout (run from mtzsyn/):
  python run_batch.py MODEL_DIR RUNS_DIR --yspec-bin Yspec/bin --table prem_sds_S.npz \
         [--phase SS] [--jobs 2] [--np 4] [--keep-raw]

For each MODEL_DIR/**/name.model:
  RUNS_DIR/name/              working folder (symlinks to yspec_pre, yspec_cal, yspec_pro, yspec_mpi)
  RUNS_DIR/name/yspec.in      written by make_yspec_in.py
  RUNS_DIR/name/post/         xsec_*.txt, summary.txt, overview.png
  raw yspec.out.* deleted after post-processing unless --keep-raw
A model whose post/summary.txt already exists is skipped, so the batch can be resumed.
Progress goes to RUNS_DIR/batch.log.
"""
import argparse, glob, os, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))


def one(model, runs_dir, yspec_bin, table, phase, nproc, keep_raw, log, timeout):
    name = os.path.splitext(os.path.basename(model))[0]
    wd = os.path.join(runs_dir, name)
    post = os.path.join(wd, "post")
    if os.path.exists(os.path.join(post, "summary.txt")):
        return name, "skip"
    os.makedirs(wd, exist_ok=True)
    for exe in ("yspec_pre", "yspec_cal", "yspec_pro", "yspec_mpi"):
        dst = os.path.join(wd, exe)
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(os.path.join(yspec_bin, exe)), dst)
    shutil.copy(model, os.path.join(wd, name + ".model"))
    t0 = time.time()
    subprocess.run([sys.executable, os.path.join(HERE, "make_yspec_in.py"),
                    name + ".model", "yspec.in", "--phase", phase],
                   cwd=wd, check=True, capture_output=True)
    for attempt in range(2):
        for f in glob.glob(os.path.join(wd, "yspec.out.*")):
            os.remove(f)
        with open(os.path.join(wd, "yspec_output"), "w") as fh:
            try:
                r = subprocess.run(["mpirun", "-n", str(nproc), "./yspec_mpi", "yspec.in"],
                                   cwd=wd, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout)
            except subprocess.TimeoutExpired:
                subprocess.run(["pkill", "-f", f"yspec_cal.*{name}"], capture_output=True)
                subprocess.run(["pkill", "-f", "yspec_cal"], capture_output=True)
                print(f"{time.strftime('%H:%M:%S')} {name} timeout, retry {attempt + 1}", file=log, flush=True)
                continue
        if r.returncode == 0 and len(glob.glob(os.path.join(wd, "yspec.out.*"))) > 0:
            break
    else:
        return name, "FAILED yspec (timeout/retries)"
    if not glob.glob(os.path.join(wd, "yspec.out.*")):
        return name, "FAILED yspec"
    r = subprocess.run([sys.executable, os.path.join(HERE, "yspec_post.py"), wd, post,
                        "--table", os.path.abspath(table), "--phase", phase],
                       capture_output=True, text=True)
    if r.returncode != 0:
        open(os.path.join(wd, "post_error.txt"), "w").write(r.stdout + r.stderr)
        return name, "FAILED post"
    if not keep_raw:
        for f in glob.glob(os.path.join(wd, "yspec.out.*")):
            os.remove(f)
    return name, f"ok {time.time() - t0:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("runs_dir")
    ap.add_argument("--yspec-bin", default="Yspec/bin")
    ap.add_argument("--table", default="prem_sds_S.npz")
    ap.add_argument("--phase", default="SS")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--np", type=int, default=4)
    ap.add_argument("--keep-raw", action="store_true")
    ap.add_argument("--timeout", type=int, default=400, help="seconds per Yspec run before it is killed and retried")
    a = ap.parse_args()

    models = sorted(glob.glob(os.path.join(a.model_dir, "**", "*.model"), recursive=True))
    os.makedirs(a.runs_dir, exist_ok=True)
    log = open(os.path.join(a.runs_dir, "batch.log"), "a")
    print(f"{len(models)} models, {a.jobs} at a time x {a.np} MPI ranks", file=log, flush=True)

    def task(m):
        name, status = one(m, a.runs_dir, a.yspec_bin, a.table, a.phase, a.np, a.keep_raw, log, a.timeout)
        print(f"{time.strftime('%H:%M:%S')} {name} {status}", file=log, flush=True)
        return status

    with ThreadPoolExecutor(a.jobs) as ex:
        results = list(ex.map(task, models))
    n_ok = sum(r.startswith("ok") for r in results)
    n_skip = sum(r == "skip" for r in results)
    print(f"done: {n_ok} ok, {n_skip} skipped, {len(results) - n_ok - n_skip} failed (see batch.log)")


if __name__ == "__main__":
    main()
