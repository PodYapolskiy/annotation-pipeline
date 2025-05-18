import os
import mlflow
import argparse
from copy import deepcopy
from pathlib import Path

# from src.description import get_description
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
            ###############
            # Description #
            ###############
            description = 'The image shows a close-up view of a washing machine in the process of being disassembled or repaired. The top lid of the washing machine is open, revealing the internal components. A basket containing some clothes (a blue item and a brown item) is placed inside the drum, which is partially visible through the open lid. The machine appears to be on a wooden floor, and there are various tools and parts around it, indicating that maintenance or repair work is being done. The brand name "Electrolux" is visible on the side of the machine.'
            # description = get_description(img_path)  # 2.5 min

            ###########
            # Objects #
            ###########
            # TODO: text -> classes (nouns)
            objects = get_objects(description)
            # objects = [
            #     "washing machine",
            #     "washing machine top lid",
            #     "basket",
            #     "clothes",
            #     "blue item",
            #     "brown item",
            #     "drum",
            # ]

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
            # bbox1 = objects[0]["bbox"]
            # bbox2 = objects[1]["bbox"]
            # print(spatial_relation(bbox1, bbox2))

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
