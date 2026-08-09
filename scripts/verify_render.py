"""Render çıktısını doğrular ve contact-sheet üretir (sistem Python + Pillow)."""
import argparse
import json
import os
from PIL import Image


def check_meta(render_dir, uid, num_views=16):
    problems = []
    d = os.path.join(render_dir, uid)
    meta_path = os.path.join(d, "meta.json")
    if not os.path.isfile(meta_path):
        return [f"{uid}: meta.json yok"]
    meta = json.load(open(meta_path))
    if meta.get("num_views") != num_views:
        problems.append(f"{uid}: num_views {meta.get('num_views')} != {num_views}")
    for i in range(num_views):
        if not os.path.isfile(os.path.join(d, f"{i:03d}.png")):
            problems.append(f"{uid}: {i:03d}.png eksik")
    return problems


def contact_sheet(render_dir, uid, out_path, cols=4):
    d = os.path.join(render_dir, uid)
    files = sorted(f for f in os.listdir(d) if f.endswith(".png"))
    imgs = [Image.open(os.path.join(d, f)).convert("RGBA") for f in files]
    w, h = imgs[0].size
    rows = (len(imgs) + cols - 1) // cols
    sheet = Image.new("RGBA", (cols * w, rows * h), (30, 30, 30, 255))
    for i, im in enumerate(imgs):
        sheet.paste(im, ((i % cols) * w, (i // cols) * h), im)
    sheet.convert("RGB").save(out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--render_dir", default="dataset/renders")
    ap.add_argument("--uid", required=True)
    ap.add_argument("--out", default="dataset/contact_sheet.png")
    a = ap.parse_args()
    probs = check_meta(a.render_dir, a.uid)
    print("SORUN YOK" if not probs else "\n".join(probs))
    contact_sheet(a.render_dir, a.uid, a.out)
    print("contact sheet:", a.out)
