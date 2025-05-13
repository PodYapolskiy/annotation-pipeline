import mlflow
import json
from pathlib import Path


def generate_json(
    img_path: Path, description: str, bboxes: list, save_dir: Path = None
) -> None:
    result = {
        "image_name": str(img_path),
        "description": description,
        "2d_bbox": bboxes,
    }

    file_name = save_dir / img_path.with_suffix(".json").name
    with open(
        file=file_name,
        mode="w",
        encoding="utf-8",
        # ...
    ) as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    mlflow.log_artifact(file_name)
