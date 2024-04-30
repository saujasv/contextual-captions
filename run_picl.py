from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from PIL import Image
import requests
import open_clip
from pathlib import Path
import os
import json
from decoding.listener import CLIPListener, Listener
from decoding.picl import PICLLogitsProcessor
from tqdm import tqdm

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b-coco")

model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b-coco",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
)

clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    model_name="ViT-B-32-quickgelu",
    pretrained="metaclip_fullcc",
    device="cuda:0",
    precision="bf16",
)
tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")
listener = CLIPListener(
    clip_model, clip_preprocess, tokenizer, "cuda:0", torch.bfloat16
)

image_path = Path("/data/tir/projects/tir3/users/svadugur/pragmatic-clip/image-sets")

for i in tqdm(range(100)):
    with open(f"./data_split/imagecode_val_shard={i}.json") as f:
        data = json.load(f)

    for x in data:
        context = [
            [Image.open(image_path / x["image_set"] / f"img{i}.jpg") for i in range(10)]
        ]
        target = int(x["image_index"])

        inputs = processor(
            images=[context[0][target]],
            return_tensors="pt",
        ).to("cuda:0")

        # autoregressively complete prompt
        output = model.generate(
            **inputs,
            max_new_tokens=32,
            logits_processor=[
                PICLLogitsProcessor(listener, context, [target], processor, 0.5, 0.01)
            ],
            do_sample=True,
            num_return_sequences=8,
            num_beams=1,
        )

        x["captions"] = processor.batch_decode(output, skip_special_tokens=True)

    with open(f"./picl_split/imagecode_val_shard={i}_captions.json", "w") as f:
        json.dump(data, f)

for i in tqdm(range(100)):
    with open(f"./data_split/imagecode_val_shard={i}.json") as f:
        data = json.load(f)

    for x in data:
        context = [
            [Image.open(image_path / x["image_set"] / f"img{i}.jpg") for i in range(10)]
        ]
        target = int(x["image_index"])

        inputs = processor(
            images=[context[0][target]],
            return_tensors="pt",
        ).to("cuda:0")

        # autoregressively complete prompt
        output = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=True,
            num_return_sequences=8,
            num_beams=1,
        )

        x["captions"] = processor.batch_decode(output, skip_special_tokens=True)

    with open(f"./ancestral_results/imagecode_val_shard={i}_captions.json", "w") as f:
        json.dump(data, f)
