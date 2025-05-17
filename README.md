# Annotation Pipeline

## Task

Учитывая что таска не риалтаймоская, а что мы собираем разметку просто, то можешь забить на временное bound и использовать модель на больше классов или вообще open dictionary

Problems:

- images with spare area, clean table
- near стиралка, не детектит бытовую технику, вещи и корзину
-

## Experiments

#### Variate detection confidence

|                       initial                       |                                  0.4                                  |                                  0.55                                   |                                  0.7                                  |
| :-------------------------------------------------: | :-------------------------------------------------------------------: | :---------------------------------------------------------------------: | :-------------------------------------------------------------------: |
| ![init](./static/variate_confidence/731_877028.jpg) | ![0.4](./static/variate_confidence/annotated_731_877028_conf_0.4.jpg) | ![0.55](./static/variate_confidence/annotated_731_877028_conf_0.55.jpg) | ![0.7](./static/variate_confidence/annotated_731_877028_conf_0.7.jpg) |

## Current Problems

#### Household Chores

|                                                                                      |                                                      |
| ------------------------------------------------------------------------------------ | ---------------------------------------------------- |
| Does not detect main object of interaction                                           | ![chores1](./static/chores/annotated_431_784902.jpg) |
| Unable to handle such an easy case for human to detect clothes in bag before laundry | ![chores2](./static/chores/548_834310.jpg)           |

## Structure

```
.
├── configs
│   └── default.yaml
├── data
├── examples
│   ├── 1.jpg
│   ├── annotated_sample.jpg
│   ├── sample.jpg
│   └── sample.json
├── scripts
│   ├── data.sh
│   ├── mlflow.sh
│   └── pipeline.sh
├── src
│   ├── __init__.py
│   ├── utils.py
│   ├── vlm.py
│   └── yolo.py
├── main.py
├── pyproject.toml
├── README.md
├── uv.lock
```

## Data

```
├── data
│   ├── processed
│   └── raw
```

Unzip archive with jpgs and place it in data/raw.

```bash
bash scripts/data.sh <.zip>
```

## Execute pipeline

```bash
uv sync
```

Serve mlflow server for tracking

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

All scripts and files should be executed from the project's root if the opposite is not explicitly stated.

## Dev

```bash
# uv sync --with dev
```

```bash
uv run pre-commit install
```
