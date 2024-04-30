from PIL import Image
import torch
import open_clip
import os
from pathlib import Path
import re
from tqdm import tqdm
import wandb
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from transformers import HfArgumentParser

RES = 224


@dataclass
class TrainingArguments:
    image_sets_path: Optional[str] = field()
    save_dir: Optional[str] = field(default=None)
    train_static_only: Optional[bool] = field(default=False)
    eval_static_only: Optional[bool] = field(default=False)
    train_batch_size: Optional[int] = field(default=36)
    eval_batch_size: Optional[int] = field(default=36)
    lr: Optional[float] = field(default=4e-6)
    model: Optional[str] = field(default="ViT-B-16")
    pretrained_ckpt: Optional[str] = field(default="openai")
    n_epochs: Optional[int] = field(default=30)
    device: Optional[str] = field(default="cuda")
    decay: Optional[float] = field(default=0.01)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    debug_steps: Optional[int] = field(default=None)
    eval_only: Optional[bool] = field(default=False)


class ImageCoDeCollator:
    def __init__(self, base_path, text_tokenizer, image_preprocessor, device="cuda"):
        self.base_path = base_path
        self.text_tokenizer = text_tokenizer
        self.image_preprocessor = image_preprocessor
        self.device = device

    def __call__(self, batch):
        img_idx = torch.tensor([int(x["image_index"]) for x in batch], dtype=torch.long)
        img_files = [
            list((Path(self.base_path) / x["image_set"]).glob("*.jpg")) for x in batch
        ]
        img_files = [
            sorted(x, key=lambda p: int(str(p).split("/")[-1].split(".")[0][3:]))
            for x in img_files
        ]
        images = [[Image.open(photo_file) for photo_file in x] for x in img_files]
        images = (
            torch.stack(
                [
                    torch.stack([self.image_preprocessor(photo) for photo in x])
                    for x in images
                ]
            )
            .to(self.device)
            .to(torch.bfloat16)
        )
        text = self.text_tokenizer([x["description"] for x in batch]).to(self.device)
        batch = {"img_idx": img_idx, "images": images, "text": text}
        return batch


def evaluate(
    model,
    tokenizer,
    image_preprocessor,
    image_sets_path,
    dataset,
    batch_size,
    device,
    debug_steps=None,
):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=ImageCoDeCollator(image_sets_path, tokenizer, image_preprocessor),
    )

    total_loss = 0
    n_samples = 0
    n_correct = 0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            if debug_steps and i > debug_steps:
                return total_loss / n_samples, n_correct / n_samples
            image_features, text_features, _ = model(
                batch["images"].view(len(batch["images"]) * 10, 3, RES, RES),
                batch["text"],
            )
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            logits_per_image = (
                model.logit_scale.exp()
                * image_features.view(len(batch["images"]), 10, -1)
                @ text_features.unsqueeze(2)
            )
            total_loss += torch.nn.functional.cross_entropy(
                logits_per_image.squeeze(), batch["img_idx"].to(device), reduction="sum"
            ).item()
            n_correct += (
                (
                    logits_per_image.argmax(dim=1).squeeze()
                    == batch["img_idx"].to(device)
                )
                .sum()
                .item()
            )
            n_samples += len(batch["images"])

    return total_loss / n_samples, n_correct / n_samples


def train(
    model,
    tokenizer,
    image_preprocessor,
    image_sets_path,
    dataset,
    train_batch_size,
    eval_batch_size,
    n_epochs,
    save_dir,
    lr=4e-6,
    decay=0,
    gradient_accumulation_steps=0,
    device="cuda",
    debug_steps=None,
):
    dataloader = DataLoader(
        dataset["train"],
        batch_size=train_batch_size,
        shuffle=False,
        collate_fn=ImageCoDeCollator(image_sets_path, tokenizer, image_preprocessor),
    )

    optimizer = optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=decay
    )
    best_loss = float("inf")
    best_ckpt = None
    for epoch in range(n_epochs):
        with torch.no_grad():
            val_loss, val_accuracy = evaluate(
                model,
                tokenizer,
                image_preprocessor,
                image_sets_path,
                dataset["validation"],
                eval_batch_size,
                device,
                debug_steps,
            )
            wandb.log({"val_accuracy": val_accuracy, "val_loss": val_loss})
            if val_loss < best_loss:
                best_loss = val_loss
                best_ckpt = epoch
                torch.save(
                    model.state_dict(), os.path.join(save_dir, f"epoch={epoch}.pt")
                )

        model.train()
        for i, batch in enumerate(tqdm(dataloader)):
            if debug_steps and i > debug_steps:
                return
            image_features_, text_features_, _ = model(
                batch["images"].view(len(batch["images"]) * 10, 3, RES, RES),
                batch["text"],
            )
            image_features = image_features_ / image_features_.norm(
                dim=-1, keepdim=True
            )
            text_features = text_features_ / text_features_.norm(dim=-1, keepdim=True)
            logits_per_image = (
                image_features.view(len(batch["images"]), 10, -1)
                @ text_features.unsqueeze(2)
            ) * model.logit_scale.exp()
            loss = torch.nn.functional.cross_entropy(
                logits_per_image.squeeze(), batch["img_idx"].to(device)
            )
            loss.backward()

            if (i + 1) % gradient_accumulation_steps == 0:
                optimizer.step()

                wandb.log({"loss": loss.item()})
                optimizer.zero_grad()


def main(args):
    wandb.init(project="pragmatic-clip")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained_ckpt, device="cuda:0", precision="bf16"
    )
    tokenizer = open_clip.get_tokenizer(args.model)

    dataset = load_dataset("BennoKrojer/ImageCoDe")
    if args.train_static_only:
        dataset["train"] = dataset["train"].filter(
            lambda x: "open-images" in x["image_set"]
        )
    if args.eval_static_only:
        dataset["validation"] = dataset["validation"].filter(
            lambda x: "open-images" in x["image_set"]
        )
        dataset["test"] = dataset["test"].filter(
            lambda x: "open-images" in x["image_set"]
        )

    if args.eval_only:
        loss, accuracy = evaluate(
            model,
            tokenizer,
            preprocess,
            args.image_sets_path,
            dataset["validation"],
            args.eval_batch_size,
            args.device,
        )
        print(f"Validation loss: {loss}, accuracy: {accuracy}")
    else:
        os.makedirs(args.save_dir, exist_ok=True)
        train(
            model,
            tokenizer,
            preprocess,
            args.image_sets_path,
            dataset,
            args.train_batch_size,
            args.eval_batch_size,
            args.n_epochs,
            args.save_dir,
            lr=args.lr,
            decay=args.decay,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            debug_steps=args.debug_steps,
        )


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments,))
    training_args = parser.parse_args_into_dataclasses()[0]
    main(training_args)
