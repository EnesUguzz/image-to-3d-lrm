"""subset.json'u gezip her objeyi Blender ile render eder.
Resume + timeout + manifest + zaman damgalı log + koşu sonu özeti + paralel worker."""
import argparse
import json
import logging
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

BLENDER_DEFAULT = r"C:\Program Files\Blender Foundation\Blender 4.4\blender.exe"


def setup_logging(output_dir):
    """Konsol + zaman damgalı dosyaya yazan logger döner."""
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    logfile = os.path.join(logs_dir, f"render_{ts}.log")
    logger = logging.getLogger("render_batch")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"log dosyasi: {logfile}")
    return logger


def _error_category(err):
    if not err:
        return None
    return "timeout" if str(err).startswith("timeout") else "render_error"


def summarize(records):
    """Bir koşunun kayıtlarından özet istatistik üretir."""
    total = len(records)
    done = sum(1 for r in records if r["status"] == "done")
    total_seconds = sum(r.get("seconds", 0) for r in records)
    slowest = sorted(records, key=lambda r: r.get("seconds", 0), reverse=True)[:5]
    error_breakdown = {}
    for r in records:
        cat = _error_category(r.get("error")) if r["status"] != "done" else None
        if cat:
            error_breakdown[cat] = error_breakdown.get(cat, 0) + 1
    return {
        "total": total,
        "done": done,
        "failed": total - done,
        "total_seconds": round(total_seconds, 1),
        "avg_seconds": round(total_seconds / total, 2) if total else 0,
        "slowest": [{"uid": r["uid"], "seconds": r.get("seconds", 0)} for r in slowest],
        "error_breakdown": error_breakdown,
    }


def _expected_views():
    """Beklenen gorunum sayisi SEMADAN gelir; sabit 16 varsayarsak sphere20 (24)
    kosusunda yarim render edilmis objeler "tamam" sayilirdi."""
    import camera_poses
    return len(camera_poses.build_view_list("_probe_"))


def is_done(output_dir, uid, num_views=None):
    num_views = _expected_views() if num_views is None else num_views
    d = os.path.join(output_dir, uid)
    if not os.path.isfile(os.path.join(d, "meta.json")):
        return False
    pngs = [f for f in os.listdir(d) if f.endswith(".png")] if os.path.isdir(d) else []
    return len(pngs) >= num_views


def render_command(blender, uid, glb_path, output_dir, resolution):
    return [blender, "-b", "--factory-startup", "-P", "scripts/render_object.py", "--",
            "--object_path", glb_path, "--output_dir", output_dir,
            "--uid", uid, "--resolution", str(resolution)]


def _render_one(uid, glb, output_dir, blender, resolution, timeout):
    """Tek objeyi render eder ve record döner. Sadece kendi obje klasörüne yazar
    (paralelde thread-safe — objeler birbirinin dosyasına dokunmaz)."""
    t0 = time.time()
    rec = {"uid": uid, "status": "done", "error": None, "blender": "4.4",
           "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
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
    return rec


def run_batch(subset, output_dir, blender=BLENDER_DEFAULT, resolution=512,
              timeout=120, workers=1):
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(output_dir)
    manifest = os.path.join(output_dir, "manifest.jsonl")
    total = len(subset)
    pending = [(uid, glb) for uid, glb in subset.items()
               if not is_done(output_dir, uid)]
    skipped = total - len(pending)
    n = len(pending)
    records = []
    lock = threading.Lock()
    counter = {"i": 0}
    t_start = time.time()
    logger.info(f"koşu başladı: {total} obje ({skipped} zaten var), "
                f"{n} render edilecek, {workers} worker, çözünürlük {resolution}")

    def task(uid, glb):
        rec = _render_one(uid, glb, output_dir, blender, resolution, timeout)
        with lock:                       # sadece defter tutma serileştirilir, render değil
            records.append(rec)
            with open(manifest, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            counter["i"] += 1
            i = counter["i"]
            if rec["status"] == "done":
                logger.info(f"[{i}/{n}] done: {uid} ({rec['seconds']}s)")
            else:
                logger.error(f"[{i}/{n}] FAILED: {uid} ({rec['seconds']}s) "
                             f"[{_error_category(rec['error'])}] {str(rec['error'])[:200]}")

    if workers <= 1:
        for uid, glb in pending:
            task(uid, glb)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for fut in [ex.submit(task, uid, glb) for uid, glb in pending]:
                fut.result()

    summary = summarize(records)
    summary["skipped"] = skipped
    summary["wall_seconds"] = round(time.time() - t_start, 1)
    with open(os.path.join(output_dir, "render_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("=" * 60)
    logger.info(f"ÖZET: {summary['done']} done, {summary['failed']} failed, "
                f"{skipped} atlandı | toplam render {summary['total_seconds']}s "
                f"(ort {summary['avg_seconds']}s/obje) | duvar saati {summary['wall_seconds']}s")
    if summary["error_breakdown"]:
        logger.info(f"HATA DÖKÜMÜ: {summary['error_breakdown']}")
    if summary["slowest"]:
        logger.info(f"EN YAVAŞ: {[(s['uid'][:8], s['seconds']) for s in summary['slowest']]}")
    logger.info(f"özet dosyası: {os.path.join(output_dir, 'render_summary.json')}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="dataset/subset.json")
    ap.add_argument("--output_dir", default="dataset/renders")
    ap.add_argument("--blender", default=BLENDER_DEFAULT)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--workers", type=int, default=1,
                    help="paralel Blender worker sayısı (VRAM ile sınırlı)")
    a = ap.parse_args()
    with open(a.subset, encoding="utf-8") as f:
        subset = json.load(f)
    run_batch(subset, a.output_dir, a.blender, a.resolution, a.timeout, a.workers)
