from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
DATA_YAML = BASE_DIR / "config.yaml"
RUN_NAME = "crater_v1_50epochs"

def main():
    # Load pretrained YOLOv8n (COCO weights) -> transfer learning
    model = YOLO("yolov8n.pt")

    model.train(
        data=str(DATA_YAML),
        epochs=50,          
        imgsz=640,
        batch=16,           
        device=0,      # change to 0 if you have a working CUDA GPU
        project=str(BASE_DIR / "runs"),
        name=RUN_NAME,
        seed=42,
    )

    # confusion matrix at a realistic confidence on the validation set, using the best weights.
    # the automatic validation during training uses conf=0.001 (for mAP), so its confusion matrix
    # counts thousands of very low-confidence boxes as false positives.
    best_model = YOLO(model.trainer.best)
    best_model.val(
        data=str(DATA_YAML),
        split="val",
        conf=0.25,
        plots=True,
        project=str(BASE_DIR / "runs"),
        name=f"{RUN_NAME}_confusion_conf025",
        exist_ok=True,
    )

if __name__ == "__main__":
    main()