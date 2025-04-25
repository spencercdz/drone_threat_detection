#!/usr/bin/env python3
# ===== AUTONOMOUS DRONE THREAT DETECTION (VisDrone-ready) =====
# Purpose : Detect potential military targets (tanks, ships, helicopters) in aerial imagery.
# Dataset : VisDrone2019-DET (local or Hugging Face subset).
#           Format: <bbox_left>,<bbox_top>,<bbox_w>,<bbox_h>,...,<class>
# Hardware: DJI Tello drone *or* any OpenCV camera.
# Author  : Updated 25 Apr 2025.

import argparse, sys, cv2, numpy as np
from pathlib import Path
from ultralytics import YOLO
from datasets import load_dataset
from ultralytics.utils.downloads import download

try:
    from djitellopy import Tello
except ImportError:
    Tello = None

# ───────────────────────── 1. INITIALISATION ──────────────────────────
def initialize_model(weights="yolov8n-obb.pt"):
    model = YOLO(weights)
    print(f"[STATUS] Model loaded ({weights}). Classes: {model.names}")
    return model

# ─────────────────── 2. VISDRONE DATASET HANDLING ─────────────────────
def ensure_visdrone_dataset(root="datasets/VisDrone"):
    root = Path(root)
    if root.exists():
        return root
    print("[INFO] VisDrone folder not found – downloading (~2.3 GB)…")
    url = "https://github.com/VisDrone/VisDrone-Dataset/releases/download/v1.0/VisDrone2019-DET.zip"
    download(url, dir=root.parent)        # Ultralytics helper
    return root

def load_visdrone_local(root="datasets/VisDrone",
                        split="train",
                        max_samples=50):
    """
    Walk VisDrone directory and yield (image, annotation) tuples.
    """
    root = ensure_visdrone_dataset(root)
    img_dir = root / f"VisDrone2019-DET-{split}/images"
    ann_dir = root / f"VisDrone2019-DET-{split}/annotations"
    if not img_dir.exists():
        raise FileNotFoundError(f"Expected {img_dir}")

    img_paths = sorted(img_dir.glob("*.jpg"))
    if max_samples > 0:
        img_paths = img_paths[:max_samples]
    samples = []
    for p in img_paths:
        ann = ann_dir / f"{p.stem}.txt"
        samples.append((str(p), str(ann)))
    print(f"[STATUS] Loaded {len(samples)} VisDrone {split} samples from {root}")
    return samples

# ─────────────────────── 3. VISUAL DEMO LOOPS ────────────────────────
def run_local_demo(model, dataset):
    """
    dataset: list of (img_path, ann_path)
    """
    for img_p, _ in dataset:
        image = cv2.imread(img_p)
        if image is None:
            continue
        results = model(image, imgsz=640)
        cv2.imshow("VisDrone Local Demo", results[0].plot())
        if cv2.waitKey(500) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()

def load_hf_dataset(n=50):
    ds = load_dataset("Voxel51/VisDrone2019-DET", split=f"train[:{n}]")
    print(f"[STATUS] Hugging Face split loaded: {len(ds)} images")
    return ds

def run_hf_demo(model, hf_ds):
    for rec in hf_ds:
        img = np.array(rec["image"])
        cv2.imshow("VisDrone HF Demo", model(img, imgsz=640)[0].plot())
        if cv2.waitKey(500) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()

# ───────────────────────── 4. LIVE DETECTION ─────────────────────────
def run_detection(model, source=0, conf=0.5):
    drone = None
    if source == "drone":
        if Tello is None:
            print("djitellopy missing.")
            sys.exit(1)
        drone = Tello(); drone.connect(); drone.streamon()
        feed = drone.get_frame_read()
    else:
        cap = cv2.VideoCapture(source)

    try:
        while True:
            frame = feed.frame if source == "drone" else cap.read()[1]
            if frame is None: break
            res = model(frame, imgsz=640, conf=conf)
            img = res[0].plot()
            for obb in res[0].obb:
                cls = model.names[int(obb.cls)]
                if cls in {"helicopter", "tank"}:
                    cv2.putText(img, f"ALERT: {cls}", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow("Threat Detection", img)
            if cv2.waitKey(1) & 0xFF == ord("q"): break
    finally:
        if source == "drone" and drone:
            drone.streamoff(); drone.end()
        elif source != "drone":
            cap.release()
        cv2.destroyAllWindows()

# ───────────────────────────── 5. CLI ────────────────────────────────
def args():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group()
    g.add_argument("--demo-hf", action="store_true", help="Run Hugging Face demo")
    g.add_argument("--demo-local", action="store_true", help="Run local VisDrone demo")
    g.add_argument("--live", action="store_true", help="Live detection (webcam/drone)")
    p.add_argument("--drone", action="store_true", help="Use DJI Tello instead of webcam")
    p.add_argument("--root", default="datasets/VisDrone", help="VisDrone root folder")
    p.add_argument("--split", default="train", help="VisDrone split (train/val/test-dev)")
    p.add_argument("--max", type=int, default=50, help="Max samples for local demo (-1 for all)")
    p.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    return p.parse_args()

def main():
    opt = args()
    model = initialize_model()

    if opt.demo_hf:
        run_hf_demo(model, load_hf_dataset())
    elif opt.demo_local:
        samples = load_visdrone_local(opt.root, opt.split, opt.max)
        run_local_demo(model, samples)
    elif opt.live:
        src = "drone" if opt.drone else 0
        run_detection(model, src, opt.conf)
    else:
        print("Choose a mode: --demo-hf  |  --demo-local  |  --live [--drone]")
        sys.exit(0)

if __name__ == "__main__":
    main()
