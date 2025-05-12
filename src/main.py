import glob
import time
import argparse
from loguru import logger

from vlm import generate_description


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input_dir", default="data/raw/")
    return parser.parse_args()


def main():
    args = parse_args()

    for img_path in glob.glob(f"{args.input_dir}/*.jpg"):
        logger.info(f"Start annotating {img_path}")

        start = time.perf_counter()
        description = generate_description(img_path)  # 2.5 min
        end = time.perf_counter()

        logger.info(f"Finished annotating {img_path} in {end-start:.2f} sec")
        logger.info(f"Description: {description}")
        break

        # image = load_image(img_path)
        # dets = detect_objects(img_path)
        # # ... остальная логика, сбор JSON
        # save_json(result, img_path.with_suffix(".json"))


if __name__ == "__main__":
    main()
