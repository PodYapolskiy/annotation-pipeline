import os
import mlflow
import argparse
from pathlib import Path

# from src.vlm import generate_description
from src.yolo import detect_objects
from src.utils import generate_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input_dir", default="data/raw/")
    parser.add_argument("--output_dir", default="data/processed/")
    parser.add_argument("--min_confidence", default=0.3)
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
            ##################
            # Bounding Boxes #
            ##################
            bboxes = detect_objects(
                img_path,
                save_dir=Path(args.output_dir),
                min_confidence=float(args.min_confidence),
            )

            ###################
            # Objects Details #
            ###################
            # TODO: ...

            ###############
            # Description #
            ###############
            description = "mock"  # generate_description(img_path)  # 2.5 min

            #####################
            # Annotation Result #
            #####################
            generate_json(
                img_path=img_path,
                description=description,
                bboxes=bboxes,
                save_dir=Path(args.output_dir),
            )

            break


if __name__ == "__main__":
    main()
