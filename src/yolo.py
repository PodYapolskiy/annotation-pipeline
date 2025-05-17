import time
import mlflow
from pathlib import Path
from loguru import logger

import cv2
from ultralytics import YOLO

# t is smallest, e is the biggest
model = YOLO("yolov9e.pt")

# attempt on YOLOWorld
# model = YOLOWorld("yolov8x-worldv2.pt")
# model.set_classes(["banana", "apple", "cart", "packet", "wires"])

# id2class = model.names


def detect_objects(
    img_path: Path, save_dir: Path = None, min_confidence: float = 0.0
) -> list[dict]:
    """
    Detects objects in an image and returns a list of dictionaries.

    Each dictionary contains the name of the detected object, its confidence,
    and its bounding box coordinates.

    If save_dir is provided, the annotated image is saved to the specified
    directory with the name prefix "annotated_".

    Args:
        img_path (Path): path to the image file
        save_dir (Path, optional): directory to save the annotated image
        min_confidence (float, optional): minimum confidence threshold for detection

    Returns:
        list[dict]: list of dictionaries containing object detection results
    """
    logger.info(f"Start bbox detection with YOLO on {img_path}")
    start = time.perf_counter()

    image = cv2.imread(str(img_path))
    results = model(image)[0]

    boxes = results.boxes
    # speed = results.speed
    # print(speed)

    detections = []
    for box in boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        x_min, y_min, x_max, y_max = map(int, box.xyxy[0])  # bbox

        if confidence >= min_confidence:
            detections.append(
                {
                    "object": class_name,
                    "confidence": round(confidence, 3),
                    "bbox": [x_min, y_min, x_max, y_max],
                }
            )

            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(
                img=image,
                text=label,
                org=(x_min, y_min - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(0, 255, 0),
                thickness=2,
            )

    if save_dir:
        cv2.imwrite(save_dir / ("annotated_" + img_path.name), image)

    end = time.perf_counter()
    logger.info(f"YOLO finished with {img_path} in {end - start:.2f} sec")
    logger.info(f"Bboxes: {detections}")
    mlflow.log_metric("bbox_detection_time", end - start)
    mlflow.log_dict(detections, f"{img_path.with_suffix('').name}_bboxes.json")

    return detections
