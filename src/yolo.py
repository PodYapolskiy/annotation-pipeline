from ultralytics import YOLO

# t is smallest, e is the biggest
model = YOLO("yolov9e.pt")


def detect_objects(img_path: str) -> list[dict]:
    results = model(img_path)

    # Extract bounding boxes from results
    bboxes = []
    for det in results[0].boxes:
        bbox = det.xyxy[0].tolist()  # Get bbox in [xmin, ymin, xmax, ymax] format
        confidence = det.conf[0].item()  # Get confidence score
        class_id = int(det.cls[0].item())  # Get class ID
        bboxes.append({"bbox": bbox, "confidence": confidence, "class_id": class_id})

    return bboxes
