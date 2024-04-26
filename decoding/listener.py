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
    
    def listener_energy(contexts: List[Image.Image], target: int,
                        processor: Blip2Processor, captioning_model: Blip2ForConditionalGeneration, 
                        input_ids: torch.Tensor, input_embeds: torch.Tensor) -> torch.Tensor:
        """Computes the log likelihood of the input_embeds given the image contexts.

        Args:
            contexts (List[Image.Image]): A list of images.
            target (int): Index of contexts to use.
            captioning_model (Blip2ForConditionalGeneration): The captioning model.
            input_embeds (torch.Tensor): The input embeddings.

        Returns:
            torch.Tensor: Log likelihoods for each context.

        Example usage:
        ```
            device = "cuda" if torch.cuda.is_available() else "cpu"
            processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b", load_in_8bit=False, device_map={"": 0}, torch_dtype=torch.float16
            )

            url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            image = Image.open(requests.get(url, stream=True).raw)
            contexts = [image, image]

            prompts = ["There are two cats in this image", "There are two dogs in this image"]
            input_ids = [processor(images=None, text=prompt, return_tensors="pt").input_ids.to(device=device) for prompt in prompts]
            input_ids = torch.cat(input_ids, dim=0)
            # TODO: Replace this with the actual embeddings
            inputs_embeds = model.language_model.get_input_embeddings()(input_ids)
            le = listener_energy(contexts, 0, processor, model, input_ids, inputs_embeds)
            le.backward()
            inputs_embeds.grad
        ```
        """
        input_embeds.retain_grad()
        pixel_values = torch.cat([processor(images=context, return_tensors="pt")['pixel_values'].to(device=device, dtype=captioning_model.dtype) for context in contexts], dim=0)
        vision_outputs = captioning_model.vision_model(
            pixel_values=pixel_values,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=False
        )
        image_embeds = vision_outputs[0]
        
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
        query_tokens = captioning_model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = captioning_model.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )
        query_output = query_outputs[0]  # (2, 32, 768)
        
        language_model_inputs = captioning_model.language_projection(query_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )
        final_inputs_embeds = torch.cat([language_model_inputs, input_embeds.to(language_model_inputs.device)], dim=1)
        
        attention_mask = torch.ones_like(input_ids)
        attention_mask = torch.cat([language_model_attention_mask, attention_mask.to(language_model_attention_mask.device)], dim=1)
        
        outputs = captioning_model.language_model(
            inputs_embeds=final_inputs_embeds,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True
        )
        
        log_probs = F.log_softmax(outputs.logits[:, 32:, :], dim=-1)
        actual_log_probs = torch.gather(log_probs, 2, input_ids.unsqueeze(-1)).squeeze(-1)
        log_likelihood = actual_log_probs.sum(dim=-1)
        return log_likelihood[target]