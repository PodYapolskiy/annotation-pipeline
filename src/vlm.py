import time
from pathlib import Path
from loguru import logger
import mlflow

import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    # attn_implementation="flash_attention_2",
)
model.eval()


# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    use_fast=True,
    # min_pixels=min_pixels,
    # max_pixels=max_pixels
)


def generate_description(img_path: Path) -> str:
    logger.info(f"Start annotating {img_path}")
    start = time.perf_counter()

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": str(img_path),
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cpu")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text: list[str] = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    end = time.perf_counter()
    logger.info(f"Finished annotating {img_path} in {end - start:.2f} sec")
    logger.info(f"Description: {output_text[0]}")
    mlflow.log_metric("description_generation_time", end - start)
    mlflow.log_text(output_text[0], "description.txt")

    return output_text[0]
