from ultralytics import YOLO

from crater_train import BASE_DIR

# load the model
model = YOLO("runs/crater_v1_50epochs/weights/best.pt")

# metrics
metrics = model.val(data="config.yaml", split="test")
print("Precision:", metrics.box.mp)
print("Recall:", metrics.box.mr)
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)
print("Time per image (ms):", metrics.speed)

# test images from the test set
model.predict("data/test/images", save=True, conf=0.25,project=str(BASE_DIR / "runs"), name="test_predictions", exist_ok=True)
