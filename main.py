#!/usr/bin/env python3
# ===== AUTONOMOUS DRONE THREAT DETECTION =====
# Purpose : Detect potential military targets (tanks, ships, helicopters) in aerial imagery.
# Hardware: DJI Tello drone *or* any OpenCV-compatible camera (e.g., Raspberry Pi Cam).
# Dataset : DOTA v2.0 (model pre-trained), VisDrone sample for quick demo.
# Notes   : Uses YOLOv8-OBB (oriented bounding boxes) from Ultralytics.

import argparse
import sys
import cv2
import numpy as np
from ultralytics import YOLO
from datasets import load_dataset

# Tello is optional; import lazily so the script also works without the SDK installed.
try:
    from djitellopy import Tello
except ImportError:
    Tello = None

# ───────────────────────── 1. INITIALIZATION ──────────────────────────
def initialize_model(weights="yolov8n-obb.pt"):
    """Load a pre-trained YOLOv8-OBB model."""
    model = YOLO(weights)
    print(f"[STATUS] Model loaded ({weights}). Classes: {model.names}")
    return model

# ──────────────────────── 2A. REAL-TIME INFERENCE ─────────────────────
def run_detection(model, source=0, conf_thresh=0.5):
    """
    Perform live inference.
    Args:
        model       : YOLOv8-OBB model.
        source      : 0 / cam path / 'drone'.
        conf_thresh : Confidence threshold (0–1).
    """
    # Choose video source
    drone = None
    if source == "drone":
        if Tello is None:
            print("ERROR: djitellopy not installed; cannot run drone mode.")
            sys.exit(1)
        drone = Tello()
        drone.connect()
        drone.streamon()
        frame_reader = drone.get_frame_read()
    else:
        cap = cv2.VideoCapture(source)

    try:
        while True:
            # Grab frame
            if source == "drone":
                frame = frame_reader.frame
            else:
                ret, frame = cap.read()
                if not ret:
                    print("Stream ended or camera not found.")
                    break

            # Inference
            results = model(frame, imgsz=640, conf=conf_thresh, verbose=False)

            # Visualize detections
            annotated = results[0].plot()
            for box in results[0].obb:          # oriented bounding boxes
                cls_id = int(box.cls)
                label = model.names[cls_id]
                if label in {"helicopter", "tank"}:        # high-priority
                    cv2.putText(
                        annotated,
                        f"ALERT: {label.upper()}",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )

            cv2.imshow("Drone Threat Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # Clean up
        if source == "drone" and drone is not None:
            drone.streamoff()
            drone.end()
        else:
            cap.release()
        cv2.destroyAllWindows()

# ─────────────────────── 2B. RASPBERRY PI SHORTCUT ────────────────────
def pi_deployment():
    """One-liner for Raspberry Pi (assuming a quantized TFLite model exists)."""
    model = initialize_model("yolov8n-obb.tflite")
    run_detection(model, source=0, conf_thresh=0.6)

# ───────────────────── 3A. HUGGING FACE DATASET DEMO ──────────────────
def load_hf_dataset(num_samples=50):
    """Pull a small VisDrone subset for a quick desktop demo."""
    ds = load_dataset("Voxel51/VisDrone2019-DET", split=f"train[:{num_samples}]")
    print(f"[STATUS] Loaded {len(ds)} samples from VisDrone")
    return ds

# ────────────────────── 3B. DATASET DEMO LOOP ─────────────────────────
def run_dataset_demo(model, dataset):
    """Display detections on sample images."""
    for item in dataset:
        img = np.array(item["image"])
        results = model(img, imgsz=640)
        cv2.imshow("Hugging Face Dataset Demo", results[0].plot())
        if cv2.waitKey(500) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()

# ─────────────────────────── 4. MAIN ──────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Autonomous Drone Threat Detection")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--demo", action="store_true", help="Run Hugging Face dataset demo")
    group.add_argument("--live", action="store_true", help="Run live detection (webcam / drone)")
    p.add_argument("--drone", action="store_true", help="Use DJI Tello instead of webcam")
    p.add_argument("--cam", type=str, default="0", help="Webcam index or video file path")
    p.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    return p.parse_args()

def main():
    args = parse_args()
    model = initialize_model()

    if args.demo:
        run_dataset_demo(model, load_hf_dataset())
    elif args.live:
        src = "drone" if args.drone else int(args.cam) if args.cam.isdigit() else args.cam
        run_detection(model, source=src, conf_thresh=args.conf)
    else:
        # Interactive menu
        choice = input("Select mode [1=dataset demo, 2=live detection]: ").strip()
        if choice == "1":
            run_dataset_demo(model, load_hf_dataset())
        else:
            src = "drone" if input("Use DJI Tello? [y/N]: ").lower().startswith("y") else 0
            run_detection(model, source=src, conf_thresh=args.conf)

if __name__ == "__main__":
    main()
