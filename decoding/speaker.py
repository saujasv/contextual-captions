from transformers import Blip2Processor, Blip2ForConditionalGeneration, Blip2ForConditionalGenerationModelOutput
from pathlib import Path
import requests
import openclip
import torchvision.transforms.transforms as T
from listener import CLIPListener, Listener
from PIL import image
import ipdb
import torch
import numpy as np


BLIP_MODEL = "Salesforce/blip2-opt-2.7b-coco"
CLIP_MODEL = "ViT-B-32-quickgelu"

N = 256
ALPHA = 1
P=3


class Blip2ForConditionalGenerationWithSampling(Blip2ForConditionalGeneration):
    config = BlipConfig

    def __init__(self, config:BlipConfig):
        super().__init__(config)
        self.config = config

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Blip2ForConditionalGenerationModelOutput]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        image_embeds = vision_outputs[1]

        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        language_model_inputs = self.language_projection(query_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )

        if input_ids is not None:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        expected_device = language_model_attention_mask.device
        attention_mask = torch.cat([language_model_attention_mask, attention_mask.to(expected_device)], dim=1)

        if self.config.use_decoder_only_language_model:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            logits = outputs.logits if return_dict else outputs[0]
            loss = None
            # we compute the loss here since we need to take into account the sequence length of the query embeds
            if labels is not None:
                labels = labels.to(logits.device)
                logits = logits[:, -labels.size(1) :, :]
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous().to(logits.device)

                # Flatten the tokens
                loss_fct = CrossEntropyLoss(reduction="mean")

                loss = loss_fct(shift_logits.view(-1, self.config.text_config.vocab_size), shift_labels.view(-1))
            else:
                outputs = self.language_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    labels=labels,
                )
                loss = outputs.loss if return_dict else outputs[0]
                logits = outputs.logits if return_dict else outputs[1]

        if not return_dict:
            outputs = (outputs[0], outputs[1], image_embeds, vision_outputs[0]) + vision_outputs[2:]
            return tuple(output for output in outputs if output is not None)

        return Blip2ForConditionalGenerationModelOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            vision_outputs=vision_outputs,
            qformer_outputs=query_outputs,
            language_model_outputs=outputs,
        )


if __name__ == "__main__":

    processor = Blip2Processor.from_pretrained(BLIP_MODEL)

    model = Blip2ForConditionalGeneration.from_pretrained(
        BLIP_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )

    url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
    image = Image.open(requests.get(url, stream=True).raw)
    prompt = ""

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        model_name=CLIP_MODEL,
        pretrained="metaclip_fulcc",
        device="cuda:1",
        precision="bfloat16",
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")
    listener = CLIPListener(clip_model, clip_preprocess, tokenizer, "cuda:1")

    image_path = Path()
    image_sets = []
    contexts = []
    targets = []

    inputs = processor(
        text=[prompt for _ in range(len(contexts))][:1]
        images=[ctx[t] for ctx, t in zip(contexts, targets)][:1]
        return_tensors="pt"
    ).to("cuda:0")

    x0 = model.text_decoder.language_model.get_input_embeddings()(inputs['input_ids'])
    x0.requires_grad = True
    inputs['input_ids'] = None

    alpha = torch.tensor(alpha, dtype=torch.FloatTensor).to("cuda:0")

    outputx0 = model(**inputs, inputs_embeds=x0)
    logitsx0 = outputx0.logits
    energyx0, = torch.autograd.grad(logitsx0, x0, torch.ones_like(logitsx0).to("cuda:0"))
    px0 = torch.exp(logitsx0)

    for n in range(N):
        eps = torch.randn(x0.shape).to("cuda:1")
        x1 = x0 - alpha / 2 * energyx0 + torch.sqrt(alpha) * eps

        outputx1 = model(**inputs, inputs_embeds=x1)
        logitsx1 = outputsx0.logits
        energyx1, = torch.autograd.grad(logitsx1, x1, torch.ones_like(logitsx1).to("cuda:0"))

        qx1x0 = torch.exp(-0.5 * torch.dot(dlogitsx0, x1 - x0) - (1 / (2 * alpha)) * torch.norm(x1 - x0, p=P))
        qx0x1 = torch.exp(-0.5 * torch.dot(dlogitsx1, x0 - x1) - (1 / (2 * alpha)) * torch.norm(x0 - x1, p=P))

        px1 = torch.exp(logitsx1)

        a = torch.min(torch.ones_like(x0).to("cuda:0"), (px1 * qx0x1) / (px0 * qx1x0))

        updates = (torch.rand(x0.shape).to("cuda:0") < a).nonzero()

        x0[updates] = x1[updates]
        outputsx0 = model(**inputs, inputs_embeds=x0)
        logitsx0 = ouptutx0.logits
        energyx0 = torch.autograd.grad(logitsx0, x0, torch.ones_like(logitsx0).to("cuda:0"))
        px0 = torch.exp(logitsx0)
        print()
        print(processor.decoder(output, skip_special_tokens=True))


