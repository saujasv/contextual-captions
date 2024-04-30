from typing import Dict, List, Union, Optional
from PIL import Image
from pathlib import Path
import torch
import json
from tqdm import tqdm
from nltk import ngrams
from evaluate import load
import numpy as np

N = 4
PERPLEXITY = load("perplexity", module_type="metric")


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


# N-gram diversity
def ngram_diversity(captions, images, index):
    unique_ngrams = [[] for _ in range(N)]
    total_ngrams = [0 for _ in range(N)]

    for c in captions:
        for n in range(N):
            g = list(ngrams(c.split(), n + 1))
            unique_ngrams[n].extend(g)
            total_ngrams[n] += len(g)

    unique_ngrams = [len(set(g)) for g in unique_ngrams]

    return np.array(
        [
            0 if total_ngrams[i] == 0 else unique_ngrams[i] / total_ngrams[i]
            for i in range(N)
        ]
    )  # / (N + 1)


# LLM perplexity of each caption - LLAMA3
def llama3_perplexity(captions, images, index):
    return PERPLEXITY.compute(predictions=captions, model_id="gpt2")["mean_perplexity"]


if __name__ == "__main__":
    from datasets import load_dataset
    from decoding.listener import CLIPListener
    import open_clip

    # dataset = (
    #     load_dataset("BennoKrojer/ImageCoDe", split="validation")
    #     .filter(lambda x: "open-images" in x["image_set"])
    #     .map(lambda x: {"captions": [x["description"] for i in range(2)]})
    #     .to_dict()
    # )

    # test_predictions = [
    #     {k: dataset[k][i] for k in dataset} for i in range(len(dataset["image_set"]))
    # ]

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16",
        pretrained="/data/tir/projects/tir3/users/svadugur/contextual-captions/siglip-ft/epoch=8.pt",
        # "laion400m_e32",
        device="cuda:0",
        precision="bf16",
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    listener = CLIPListener(
        model, preprocess, tokenizer, device="cuda:0", dtype=torch.bfloat16
    )

    evaluation_functions = {
        "listener_success": listener.evaluate_success,
        "ngram_diversity": ngram_diversity,
        "perplexity": llama3_perplexity,
    }

    with open("results_pooled/pncg.json", "r") as f:
        test_predictions = json.load(f)

    metrics = evaluate(
        test_predictions,
        evaluation_functions,
        "/data/tir/projects/tir3/users/svadugur/pragmatic-clip/image-sets",
    )

    print(f"pNCG results ({len(metrics)}):")
    print("Listener success:", np.mean([m["listener_success"] for m in metrics]))
    print(
        "n-gram diversity",
        np.stack([m["ngram_diversity"] for m in metrics]).mean(axis=0).tolist(),
    )
    print("Perplexity:", np.mean([m["perplexity"] for m in metrics]))

    with open("results_pooled/picl.json", "r") as f:
        test_predictions = json.load(f)

    metrics = evaluate(
        test_predictions,
        evaluation_functions,
        "/data/tir/projects/tir3/users/svadugur/pragmatic-clip/image-sets",
    )

    print(f"PICL results ({len(metrics)}):")
    print("Listener success:", np.mean([m["listener_success"] for m in metrics]))
    print(
        "n-gram diversity",
        np.stack([m["ngram_diversity"] for m in metrics]).mean(axis=0).tolist(),
    )
    print("Perplexity:", np.mean([m["perplexity"] for m in metrics]))

    with open("results_pooled/top_p.json", "r") as f:
        test_predictions = json.load(f)

    metrics = evaluate(
        test_predictions,
        evaluation_functions,
        "/data/tir/projects/tir3/users/svadugur/pragmatic-clip/image-sets",
    )

    print(f"Top-p results ({len(metrics)}):")
    print("Listener success:", np.mean([m["listener_success"] for m in metrics]))
    print(
        "n-gram diversity",
        np.stack([m["ngram_diversity"] for m in metrics]).mean(axis=0).tolist(),
    )
    print("Perplexity:", np.mean([m["perplexity"] for m in metrics]))
