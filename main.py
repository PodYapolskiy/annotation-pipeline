import os
import mlflow
import argparse
from copy import deepcopy
from pathlib import Path

from src.description import get_description
from src.objects import get_objects
from src.detection import get_detections
from src.utils import generate_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input_dir", default="data/raw/")
    parser.add_argument("--output_dir", default="data/processed/")
    parser.add_argument("--min_confidence", default=0.01)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs("mlflow", exist_ok=True)
    os.makedirs(args.input_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    mlflow.set_tracking_uri(uri="http://localhost:5000")
    mlflow.set_experiment("annotation-pipeline")

    with mlflow.start_run():
        for img_path in Path(args.input_dir).glob("*.jpg"):
            # skip annotated if same folder is used
            if img_path.name.startswith("annotated_"):
                continue

            ###############
            # Description #
            ###############
            description = get_description(img_path)  # +-2.5 min on cpu

            ###########
            # Objects #
            ###########
            objects = get_objects(description)

            ##############
            # Detections #
            ##############
            detections = get_detections(
                img_path=img_path,
                save_dir=Path(args.output_dir),
                classes=objects,
                min_confidence=float(args.min_confidence),
            )
            bboxes = deepcopy(detections)

            ###################
            # Objects Details #
            ###################
            # TODO: ...

            #####################
            # Annotation Result #
            #####################
            generate_json(
                img_path=img_path,
                description=description,
                bboxes=bboxes,
                save_dir=Path(args.output_dir),
            )


if __name__ == "__main__":
    main()
