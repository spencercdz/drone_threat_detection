# ===== AUTONOMOUS DRONE THREAT DETECTION =====
# Purpose: Detect military targets (tanks, ships, helicopters) in aerial imagery using YOLOv8-OBB
# Hardware: Works with DJI Tello drone or Raspberry Pi + Pi Camera
# Dataset: DOTA v2.0 (preprocessed for YOLO-OBB format)
# Dataset: Hugging Face dataset integration for demo purposes


import cv2
import numpy as np
from ultralytics import YOLO
from djitellopy import Tello  # For drone control (optional)
from datasets import load_dataset

# ===== 1. INITIALIZATION =====
def initialize_models():
    """Load pre-trained YOLOv8-OBB model (pre-trained on DOTA)"""
    model = YOLO('yolov8n-obb.pt')  # Nano version for edge devices
    print("[STATUS] Model loaded. Classes:", model.names)
    return model

# ===== 2A. FOR REAL: REAL-TIME INFERENCE =====
def run_detection(model, source=0, conf_thresh=0.5):
    """
    Run inference on live stream (drone/camera)
    
    Args:
        model: YOLOv8-OBB model
        source: 0 for webcam, 'drone' for Tello, or video path
        conf_thresh: Confidence threshold (0.25-0.7)
    """
    # Initialize video source
    if source == 'drone':
        drone = Tello()
        drone.connect()
        drone.streamon()
        cap = drone.get_frame_read()
    else:
        cap = cv2.VideoCapture(source)  # 0 for Pi Camera
    
    while True:
        # Get frame
        if source == 'drone':
            frame = cap.frame
        else:
            ret, frame = cap.read()
            if not ret: break
        
        # Run inference (oriented bounding boxes)
        results = model(frame, imgsz=640, conf=conf_thresh, verbose=False)
        
        # Visualize results
        annotated_frame = results[0].plot()  # Draw rotated boxes
        
        # Display threat alerts
        for box in results[0].obb:
            class_id = int(box.cls)
            label = model.names[class_id]
            if label in ['helicopter', 'tank']:  # High-priority targets
                cv2.putText(annotated_frame, "ALERT: " + label, (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Show output
        cv2.imshow('Drone Threat Detection', annotated_frame)
        if cv2.waitKey(1) == ord('q'):
            break
    
    if source == 'drone':
        drone.streamoff()
    else:
        cap.release()
    cv2.destroyAllWindows()

# ===== 2B. FOR REAL: DEPLOYMENT ON RASPBERRY PI =====
def pi_deployment():
    """Optimized version for Raspberry Pi with Pi Camera"""
    model = YOLO('yolov8n-obb.tflite')  # Quantized TensorFlow Lite model
    run_detection(model, source=0, conf_thresh=0.6)  # Higher threshold to reduce false positives

# ===== 3A. FOR DEMO: LOAD HUGGING FACE DATASET =====
def load_hf_dataset():
    """Load VisDrone dataset from Hugging Face (aerial military-relevant objects)"""
    dataset = load_dataset("Voxel51/VisDrone2019-DET", split="train[:50]")  # First 50 samples
    print(f"[STATUS] Loaded {len(dataset)} samples from VisDrone")
    return dataset

# ===== 3B. DATASET DEMO MODE =====
def run_dataset_demo(model, dataset):
    """Run detection on Hugging Face dataset samples"""
    for sample in dataset:
        image = np.array(sample['image'])
        results = model(image, imgsz=640)
        
        # Visualize with annotations
        annotated = results[0].plot()
        cv2.imshow('Hugging Face Dataset Demo', annotated)
        if cv2.waitKey(500) == ord('q'):  # 0.5s delay between samples
            break
    cv2.destroyAllWindows()

# ===== 4. MAIN EXECUTION =====
if __name__ == "__main__":
    # Initialize
    model = initialize_models()
    hf_dataset = load_hf_dataset()
    
    # Choose mode
    print("Select mode:")
    print("1 - Hugging Face Dataset Demo")
    print("2 - Live Drone Detection")
    
    choice = input("Enter choice (1/2): ")
    
    if choice == "1":
        run_dataset_demo(model, hf_dataset)
    else:
        run_detection(model, source='drone')