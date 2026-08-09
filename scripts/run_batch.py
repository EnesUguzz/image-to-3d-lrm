"""subset.json'u gezip her objeyi Blender ile render eder. Resume + timeout + manifest."""
import argparse
import json
import os
import subprocess
import time

BLENDER_DEFAULT = r"C:\Program Files\Blender Foundation\Blender 4.4\blender.exe"


def is_done(output_dir, uid, num_views=16):
    d = os.path.join(output_dir, uid)
    if not os.path.isfile(os.path.join(d, "meta.json")):
        return False
    pngs = [f for f in os.listdir(d) if f.endswith(".png")] if os.path.isdir(d) else []
    return len(pngs) >= num_views


def render_command(blender, uid, glb_path, output_dir, resolution):
    return [blender, "-b", "--factory-startup", "-P", "scripts/render_object.py", "--",
            "--object_path", glb_path, "--output_dir", output_dir,
            "--uid", uid, "--resolution", str(resolution)]


def run_batch(subset, output_dir, blender=BLENDER_DEFAULT, resolution=512, timeout=120):
    os.makedirs(output_dir, exist_ok=True)
    manifest = os.path.join(output_dir, "manifest.jsonl")
    total = len(subset)
    for i, (uid, glb) in enumerate(subset.items(), 1):
        if is_done(output_dir, uid):
            print(f"[{i}/{total}] atla (tamam): {uid}")
            continue
        t0 = time.time()
        rec = {"uid": uid, "status": "done", "error": None, "blender": "4.4"}
        try:
            r = subprocess.run(render_command(blender, uid, glb, output_dir, resolution),
                               capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=timeout)
            if "RENDER_OK" not in r.stdout or not is_done(output_dir, uid):
                rec["status"] = "failed"
                rec["error"] = (r.stdout + r.stderr)[-500:]
        except subprocess.TimeoutExpired:
            rec["status"] = "failed"
            rec["error"] = f"timeout>{timeout}s"
        rec["seconds"] = round(time.time() - t0, 1)
        with open(manifest, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{i}/{total}] {rec['status']}: {uid} ({rec['seconds']}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="dataset/subset.json")
    ap.add_argument("--output_dir", default="dataset/renders")
    ap.add_argument("--blender", default=BLENDER_DEFAULT)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--timeout", type=int, default=120)
    a = ap.parse_args()
    with open(a.subset) as f:
        subset = json.load(f)
    run_batch(subset, a.output_dir, a.blender, a.resolution, a.timeout)
