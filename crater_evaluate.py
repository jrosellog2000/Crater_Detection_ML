import cv2
import numpy as np
from ultralytics import YOLO

from crater_train import BASE_DIR

# load the model
model = YOLO("runs/crater_v1_50epochs/weights/best.pt")

# metrics
metrics = model.val(
    data="config.yaml",
    split="test",
    plots=True,
    project=str(BASE_DIR / "runs"),
    name="test_evaluation",
    exist_ok=True,
)
print("Precision:", metrics.box.mp)
print("Recall:", metrics.box.mr)
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)
print("Time per image (ms):", metrics.speed)

# second validation only for the confusion matrix: the run above keeps every box down to conf=0.001
# (needed for a correct mAP), which fills the confusion matrix with thousands of faint "false positives".
# Using conf=0.25 here matches what the prediction images show.
model.val(
    data="config.yaml",
    split="test",
    conf=0.25,
    plots=True,
    project=str(BASE_DIR / "runs"),
    name="test_confusion_conf025",
    exist_ok=True,
)

# join each real-labels grid with its prediction grid: real labels on the left, predictions on the right
eval_dir = metrics.save_dir
for labels_path in sorted(eval_dir.glob("val_batch*_labels.jpg")):
    pred_path = labels_path.with_name(labels_path.name.replace("_labels", "_pred"))
    if not pred_path.exists():
        continue
    labels_img = cv2.imread(str(labels_path))
    pred_img = cv2.imread(str(pred_path))
    pred_img = cv2.resize(pred_img, (labels_img.shape[1], labels_img.shape[0]))
    # add a black header bar on top of each grid and write the title there
    labels_img = cv2.copyMakeBorder(labels_img, 60, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    pred_img = cv2.copyMakeBorder(pred_img, 60, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.putText(labels_img, "REAL LABELS", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
    cv2.putText(pred_img, "PREDICTIONS", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
    out_path = labels_path.with_name(labels_path.name.replace("_labels", "_comparison"))
    cv2.imwrite(str(out_path), np.hstack([labels_img, pred_img]))
    print("Comparison saved:", out_path)

# test images from the test set
model.predict("data/test/images", save=True, conf=0.25, project=str(BASE_DIR / "runs"), name="test_predictions", exist_ok=True)
