from typing import List, Union, Optional
from pathlib import Path
import open_clip
import torch
import torchvision.transforms.transforms as T
from PIL import Image
from einops import rearrange
import itertools


class Listener:
    def encode_images(self, images: Union[List[Path], List[str]]):
        raise NotImplementedError

    def encode_texts(self, texts: List[str]):
        raise NotImplementedError

    def score_texts(
        self,
        texts: List[str],
        images: Optional[Union[List[str], List[Path]]] = None,
        image_features: Optional[torch.Tensor] = None,
    ):
        raise NotImplementedError


class CLIPListener(Listener):
    def __init__(
        self,
        model: open_clip.model.CLIP,
        image_processor: T.Compose,
        tokenizer: Union[
            open_clip.tokenizer.SimpleTokenizer,
            open_clip.tokenizer.HFTokenizer,
            open_clip.tokenizer.SigLipTokenizer,
        ],
        device,
    ):
        super().__init__()
        self.model = model
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.device = device

    def encode_images(
        self, images: List[List[Union[List[Path], List[str], List[Image.Image]]]]
    ):
        if isinstance(images[0][0], str) or isinstance(images[0][0], Path):
            image_inputs = torch.stack(
                list(
                    itertools.chain.from_iterable(
                        [
                            [
                                self.image_processor(Image.open(image))
                                for image in context
                            ]
                            for context in images
                        ]
                    )
                )
            ).to(self.device, dtype=torch.bfloat16)
        else:
            image_inputs = torch.stack(
                list(
                    itertools.chain.from_iterable(
                        [
                            [self.image_processor(image) for image in context]
                            for context in images
                        ]
                    )
                )
            ).to(self.device, dtype=torch.bfloat16)

        image_features = self.model.encode_image(image_inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return rearrange(image_features, "(b n) d -> b n d", b=len(images))

    def encode_texts(self, texts: List[str]):
        text_inputs = self.tokenizer(texts).to(self.device)
        text_features = self.model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    @torch.no_grad()
    def score_texts(
        self,
        texts: List[str],
        images: Optional[Union[List[str], List[Path]]] = None,
        image_features: Optional[torch.Tensor] = None,
    ):
        if image_features is None:
            if images is None:
                raise ValueError("Either images or image_embeddings must be provided")
            image_embeddings = self.encode_images(images)
        else:
            image_embeddings = image_features

        text_features = self.encode_texts(texts)
        text_probs = (
            self.model.logit_scale * image_embeddings @ text_features.T
        ).log_softmax(dim=-2)

        return text_probs

    @torch.no_grad()
    def evaluate_success(
        self,
        texts: List[str],
        images: Union[List[str], List[Path]],
        targets: Union[List[int], int],
    ):
        if isinstance(targets, int):
            targets = [targets for _ in texts]

        text_probs = self.score_texts(texts, [images])
        predictions = text_probs.argmax(dim=-2)

        return (
            (
                predictions
                == torch.tensor(targets, dtype=torch.long, device=predictions.device)
            )
            .float()
            .mean()
            .item()
        )
    
    def compute_gradients(
        self,
        text_probs: torch.Tensor,
        text_input_ids: torch.Tensor,
    ):
        # Ensure targets are a tensor and on the same device as text_probs
        targets = torch.arange(text_probs.size(0), device=text_probs.device)
        # One hot encode targets
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=text_probs.size(1)).float()

        # Compute loss with respect to the targets
        loss = torch.nn.functional.binary_cross_entropy(text_probs, targets_one_hot, reduction='sum')
        loss.backward()  # Compute gradients with respect to text inputs

        return text_input_ids.grad  # Return gradients
