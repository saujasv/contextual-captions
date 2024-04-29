from typing import List, Union, Optional
from pathlib import Path
import open_clip
import torch
import torchvision.transforms.transforms as T
from PIL import Image
from einops import rearrange
import itertools
from transformers import Blip2ForConditionalGeneration, Blip2Processor
import torch.nn.functional as F
from .utils import get_input_ids


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


class Blip2Listener:
    def __init__(self, model: Blip2ForConditionalGeneration, processor: Blip2Processor):
        self.model = model
        self.processor = processor

    def get_input_ids(self, input_embeds: torch.Tensor):
        # Solution due to https://discuss.pytorch.org/t/reverse-nn-embedding/142623/8
        embeddings = self.model.language_model.get_input_embeddings().weight

        return get_input_ids(embeddings, input_embeds)

    def energy(
        self,
        input_embeds: torch.Tensor,
        contexts: List[Image.Image],
        target: int,
    ) -> torch.Tensor:
        input_ids = self.get_input_ids(input_embeds)

        pixel_values = torch.cat(
            [
                self.processor(images=context, return_tensors="pt")["pixel_values"].to(
                    device=self.model.device, dtype=self.model.dtype
                )
                for context in contexts
            ],
            dim=0,
        )
        vision_outputs = self.model.vision_model(
            pixel_values=pixel_values,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=False,
        )
        image_embeds = vision_outputs[0]

        image_attention_mask = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device
        )
        query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.model.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )
        query_output = query_outputs[0]  # (2, 32, 768)

        language_model_inputs = self.model.language_projection(query_output)

        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1],
            dtype=torch.long,
            device=language_model_inputs.device,
        )
        final_inputs_embeds = torch.cat(
            [
                language_model_inputs,
                input_embeds.expand((language_model_inputs.size(0), -1, -1)).to(
                    language_model_inputs.device
                ),
            ],
            dim=1,
        )

        attention_mask = torch.ones_like(input_ids)
        attention_mask = torch.cat(
            [
                language_model_attention_mask,
                attention_mask.expand((language_model_inputs.size(0), -1)).to(
                    language_model_attention_mask.device
                ),
            ],
            dim=1,
        )

        outputs = self.model.language_model(
            inputs_embeds=final_inputs_embeds,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        log_probs = F.log_softmax(
            outputs.logits[:, language_model_inputs.size(1) :, :], dim=-1
        )
        actual_log_probs = torch.gather(
            log_probs,
            2,
            input_ids.expand((language_model_inputs.size(0), -1)).unsqueeze(-1),
        ).squeeze(-1)
        log_likelihood = actual_log_probs.sum(dim=-1)

        return -log_likelihood[target] + torch.logsumexp(log_likelihood, 0)


if __name__ == "__main__":
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    import torch
    from PIL import Image
    from pathlib import Path

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b-coco")

    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b-coco",
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
    )

    listener = Blip2Listener(model, processor)

    img_set = "open-images-2057_2fc6afbbb663b164"
    image_path = Path(
        "/data/tir/projects/tir3/users/svadugur/pragmatic-clip/image-sets"
    )
    images = [Image.open(image_path / img_set / f"img{i}.jpg") for i in range(10)]

    text = "A elderly lady in a pink top is having a conversation with another elderly lady."
    input_ids = processor(text=text, return_tensors="pt").input_ids.to(model.device)
    print(input_ids)

    embeds = model.language_model.get_input_embeddings()(input_ids)
    print(listener.get_input_ids(embeds))

    grad_energy = torch.func.grad(listener.energy)
    print(grad_energy.input_size)
    # (embeds, images, 5)
    # print(torch.einsum("bid,bid->bi", grad_energy, embeds).shape)
