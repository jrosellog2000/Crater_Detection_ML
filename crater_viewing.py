from pathlib import Path
import cv2
from ultralytics import YOLO

# load model
model = YOLO("yolov8n.pt")

#load image 
image_path = (
    Path(__file__).resolve().parent
    / "data"
    / "train"
    / "images"
    / "01_png.rf.4d2ebc5ed98ad1e69d667aadbce63d53.jpg"
)
img = cv2.imread(str(image_path))
cv2.imshow("crater", img)
cv2.waitKey(0)