# Annotation Pipeline

## Task

TODO: ...

<!-- Учитывая что таска не риалтаймоская, а что мы собираем разметку просто, то можешь забить на временное bound и использовать модель на больше классов или вообще open dictionary -->

## Solution / Pipeline

TODO: mermaid

## Experiments

### YOLO

Variate detection confidence

|                       initial                       |                                  0.4                                  |                                  0.55                                   |                                  0.7                                  |
| :-------------------------------------------------: | :-------------------------------------------------------------------: | :---------------------------------------------------------------------: | :-------------------------------------------------------------------: |
| ![init](./static/variate_confidence/731_877028.jpg) | ![0.4](./static/variate_confidence/annotated_731_877028_conf_0.4.jpg) | ![0.55](./static/variate_confidence/annotated_731_877028_conf_0.55.jpg) | ![0.7](./static/variate_confidence/annotated_731_877028_conf_0.7.jpg) |

### YOLO Problems

Main default YOLO problems are the lack of support for rare objects classes that would frequently appear in free environment of a robot.

The minimum confidence for all images here is 0.0, meaning that YOLO even don't try to detect them.

|                                                                                      |                                                      |
| ------------------------------------------------------------------------------------ | ---------------------------------------------------- |
| Does not detect main object of interaction                                           | ![chores1](./static/chores/annotated_431_784902.jpg) |
| Unable to handle such an easy case for human to detect clothes in bag before laundry | ![chores2](./static/chores/548_834310.jpg)           |

### YOLOWorld

![YOLOWorld](https://github.com/ultralytics/docs/releases/download/0/yolo-world-model-architecture-overview.avif)

Changing detection model to [YOLOWorld](https://docs.ultralytics.com/models/yolo-world/) helped a lot due to its nature to find default classes using CLIP text encoder to embed custom classes and further detect them on image.

|                        initial                         | YOLOWorld                                                            |
| :----------------------------------------------------: | -------------------------------------------------------------------- |
| ![init](./static/yolo-world-confidence/548_834310.jpg) | ![0.3](./static/yolo-world-confidence/annotated_548_834310_0.03.jpg) |

### YOLOWorld Problems

One of the firstly observed problems is much lower minimum confidence level needed for detecting custom classes. So, here number of detected bounding boxes for provided classes `["open through the open lid", "a basket", "a close-up view", "item some clothes"]` are changing a lot after changing even on `0.01`.

|                        initial                         |                                 0.01                                  |                                 0.02                                  |                                 0.03                                 |
| :----------------------------------------------------: | :-------------------------------------------------------------------: | :-------------------------------------------------------------------: | :------------------------------------------------------------------: |
| ![init](./static/yolo-world-confidence/548_834310.jpg) | ![0.01](./static/yolo-world-confidence/annotated_548_834310_0.01.jpg) | ![0.02](./static/yolo-world-confidence/annotated_548_834310_0.02.jpg) | ![0.3](./static/yolo-world-confidence/annotated_548_834310_0.03.jpg) |

Calling other problems it should be mentioned that for such a small probabalities there are sometimes repeated bounding boxes of the same object.

|                                                                                                                                                        |                                                                       |
| ------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------- |
| _"open through the open lid"_ bboxes refers to the same object (`Non-Maximum Suppression` is needed)                                                   | ![nms](./static/yolo-world-confidence/annotated_548_834310_0.03.jpg)  |
| You may see a bbox around the picture which is of class _"a close-up view"_ which definetely should not be considered. (`Filtering by area` is needed) | ![area](./static/yolo-world-confidence/annotated_548_834310_0.03.jpg) |

## Structure

```
.
├── configs
│   └── default.yaml
├── data
│   ├── processed
│   └── raw
├── examples
│   ├── annotated_sample.jpg
│   ├── sample.jpg
│   └── sample.json
├── scripts
│   ├── data.sh
│   ├── mlflow.sh
│   └── pipeline.sh
├── src
│   ├── description.py
│   ├── detection.py
│   ├── objects.py
│   └── utils.py
├── main.py
├── pyproject.toml
├── README.md
├── uv.lock
```

> [!WARNING]
> All scripts should be executed from the project's root if the opposite is not explicitly stated

## Data

```
├── data
│   ├── processed
│   └── raw
```

Use script to unzip archive with jpgs and place it in data/raw.

```bash
bash scripts/data.sh <.zip>
```

## Execute pipeline

```bash
uv sync
```

Serve mlflow for experiments tracking

```bash
bash scripts/mlflow.sh
```

Execute the annotation pipeline itself

```bash
bash scripts/pipeline.sh
```

## Test on one sample

To test on 1 sample use

```bash
uv run main.py --input_dir examples --output_dir examples
```

## Dev

Includes ruff, pre-commit utilites

```bash
uv sync --dev
```

```bash
uvx pre-commit install
```
