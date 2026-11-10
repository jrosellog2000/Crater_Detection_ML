from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
DATA_YAML = BASE_DIR / "config.yaml"   

def main():
    # Load pretrained YOLOv8n (COCO weights) -> transfer learning
    model = YOLO("yolov8n.pt")

    model.train(
        data=str(DATA_YAML),
        epochs=5,          
        imgsz=640,
        batch=8,           # lower it if you run out of memory
        device=0,      # change to 0 if you have a working CUDA GPU
        project=str(BASE_DIR / "runs"),
        name="crater_test",
        seed=42,
    )

if __name__ == "__main__":
    main()