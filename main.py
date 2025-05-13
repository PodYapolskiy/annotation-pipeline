import time
import argparse
from pathlib import Path
from loguru import logger

from src.vlm import generate_description
from src.yolo import detect_objects
from src.utils import generate_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input_dir", default="data/raw/")
    parser.add_argument("--output_dir", default="data/processed/")
    return parser.parse_args()


def main():
    args = parse_args()

    for img_path in Path(args.input_dir).glob("*.jpg"):
        ###############
        # Description #
        ###############
        logger.info(f"Start annotating {img_path}")
        start = time.perf_counter()
        description = "mock"  # generate_description(img_path)  # 2.5 min
        end = time.perf_counter()
        logger.info(f"Finished annotating {img_path} in {end - start:.2f} sec")
        logger.info(f"Description: {description}")

        ##################
        # Bounding Boxes #
        ##################
        logger.info(f"Start bbox detection with YOLO on {img_path}")
        start = time.perf_counter()
        bboxes = []  # detect_objects(img_path)
        end = time.perf_counter()
        logger.info(f"YOLO finished with {img_path} in {end - start:.2f} sec")
        logger.info(f"Bboxes: {bboxes}")

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
