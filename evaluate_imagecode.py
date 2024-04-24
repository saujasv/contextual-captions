from typing import Dict, List, Union, Optional
from PIL import Image
from pathlib import Path
import torch
import json
from tqdm import tqdm
import ssl
ssl._create_default_https_context = ssl._create_unverified_context  # Workaround for SSL error


def evaluate(
    predicted_captions: List[dict],
    evaluation_functions: Dict[str, callable],
    images_path: Union[str, Path],
):
    images_path = Path(images_path) if isinstance(images_path, str) else images_path
    metrics = []
    for x in tqdm(predicted_captions, desc="Evaluation"):
        images = [
            Image.open(images_path / x["image_set"] / f"img{i}.jpg") for i in range(10)
        ]
        metrics.append(
            {
                metric_name: metric(
                    x["captions"],
                    images,
                    int(x["image_index"]),
                )
                for metric_name, metric in evaluation_functions.items()
            }
        )
        torch.cuda.empty_cache()

    return metrics


if __name__ == "__main__":
    from datasets import load_dataset
    from decoding.listener import CLIPListener
    import open_clip

    dataset = (
        load_dataset("BennoKrojer/ImageCoDe", split="validation")
        .filter(lambda x: "open-images" in x["image_set"])
        .map(lambda x: {"captions": [x["description"] for i in range(2)]})
        .to_dict()
    )

    test_predictions = [
        {k: dataset[k][i] for k in dataset} for i in range(len(dataset["image_set"]))
    ]

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", device="cuda:0", precision="bf16"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    listener = CLIPListener(model, preprocess, tokenizer, device="cuda:0")

    evaluation_functions = {"listener_success": listener.evaluate_success}

    metrics = evaluate(
        test_predictions,
        evaluation_functions,
        "/data/tir/projects/tir3/users/svadugur/pragmatic-clip/image-sets",
    )
    print(metrics)
