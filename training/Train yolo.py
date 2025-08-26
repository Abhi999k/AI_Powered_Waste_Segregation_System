
import os
from ultralytics import YOLO

dataset_dir = r"detection dataset"
yaml_path = os.path.join(dataset_dir, "data.yaml")

with open(yaml_path, "r") as f:
    print(f.read())

# Load model
model = YOLO("yolov8n.pt")  

# Train
model.train(
    data=yaml_path,
    epochs=50,
    imgsz=640,
    batch=16,
    project="yolo_training_results",
    name="waste_detection",
)
