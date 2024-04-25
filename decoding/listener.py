from typing import List, Union, Optional
from pathlib import Path
import open_clip
from open_clip.transformer import text_global_pool
import torch
from torch import nn
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
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return rearrange(image_features, "(b n) d -> b n d", b=len(images))

    def encode_texts(self, texts: List[str]):
        text_inputs = self.tokenizer(texts).to(self.device)
        cast_dtype = self.model.transformer.get_cast_dtype()

        text_features = self.model.token_embedding(text_inputs).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        text_features = text_features + self.model.positional_embedding.to(cast_dtype)
        text_features = text_features.permute(1, 0, 2)  # NLD -> LND
        text_features = self.model.transformer(text_features, attn_mask=self.model.attn_mask)
        text_features = text_features.permute(1, 0, 2)  # LND -> NLD
        text_features = self.model.ln_final(text_features)  # [batch_size, n_ctx, transformer.width]

        pooled_text_features, _ = text_global_pool(text_features, text_inputs, self.model.text_pool_type)
        if self.model.text_projection is not None:
            if isinstance(self.model.text_projection, nn.Linear):
                pooled_text_features = self.model.text_projection(pooled_text_features)
            else:
                pooled_text_features = pooled_text_features @ self.model.text_projection

        pooled_text_features = pooled_text_features / pooled_text_features.norm(dim=-1, keepdim=True)
        return pooled_text_features, text_features

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

        pooled_text_features, _ = self.encode_texts(texts)
        text_probs = (
            self.model.logit_scale * image_embeddings @ pooled_text_features.T
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
