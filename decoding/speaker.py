from pathlib import Path
from typing import List
import torch.nn.functional as F
from PIL import Image
from utils import get_input_ids
import torch


BLIP_MODEL = "Salesforce/blip2-opt-2.7b-coco"
CLIP_MODEL = "ViT-B-32-quickgelu"

N = 256
ALPHA = 1
P = 3


class Blip2Speaker:
    def __init__(self, model, processor):
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

        # input_embeds.retain_grad()
        pixel_values = self.processor(images=contexts[target], return_tensors="pt")[
            "pixel_values"
        ].to(device=self.model.device, dtype=self.model.dtype)
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
        log_likelihood = actual_log_probs.sum()

        return log_likelihood


class GPT2Speaker:
    def __init__(self, model, processor):
        self.model = model
        self.model.eval()
        self.processor = processor

    def get_input_ids(self, input_embeds: torch.Tensor):
        # Solution due to https://discuss.pytorch.org/t/reverse-nn-embedding/142623/8
        embeddings = self.model.get_input_embeddings().weight

        return get_input_ids(embeddings, input_embeds)

    def energy(
        self,
        input_embeds: torch.Tensor,
        contexts,
        target,
    ) -> torch.Tensor:
        input_ids = self.get_input_ids(input_embeds)

        embeds_with_bos = torch.cat(
            (
                self.model.get_input_embeddings()(
                    torch.tensor(
                        self.processor.bos_token_id,
                        dtype=torch.long,
                        device=self.model.device,
                    )
                )
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(input_embeds.shape[0], 1, -1),
                input_embeds,
            ),
            dim=1,
        )

        self.model.eval()
        outputs = self.model(
            inputs_embeds=embeds_with_bos,
            # inputs_embeds=input_embeds,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            # labels=torch.cat(
            #     (
            #         torch.tensor(
            #             self.processor.bos_token_id,
            #             dtype=torch.long,
            #             device=input_ids.device,
            #         )
            #         .unsqueeze(0)
            #         .unsqueeze(0),
            #         input_ids,
            #     ),
            #     dim=1,
            # ),
        )

        log_probs = F.log_softmax(outputs.logits, dim=-1)
        # actual_log_probs = torch.gather(log_probs, 2, input_ids.unsqueeze(2)).squeeze(
        #    -1
        #)
        #log_likelihood = actual_log_probs.sum(dim=-1)

        actual_log_probs = torch.gather(
            log_probs[:, :-1, :], 2, input_ids.unsqueeze(2)
        ).squeeze(-1)
        #print(actual_log_probs.shape)
        #print(actual_log_probs)
        log_likelihood = actual_log_probs.sum(dim=-1)

        return -log_likelihood.squeeze()


if __name__ == "__main__":
    # from transformers import Blip2Processor, Blip2ForConditionalGeneration
    # import torch
    # from PIL import Image
    # from pathlib import Path

    # processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b-coco")

    # model = Blip2ForConditionalGeneration.from_pretrained(
    #     "Salesforce/blip2-opt-2.7b-coco",
    #     device_map="cuda:0",
    #     torch_dtype=torch.bfloat16,
    # )

    # speaker = Blip2Speaker(model, processor)

    # img_set = "open-images-2057_2fc6afbbb663b164"
    # image_path = Path(
    #     "/data/tir/projects/tir3/users/svadugur/pragmatic-clip/image-sets"
    # )
    # images = [Image.open(image_path / img_set / f"img{i}.jpg") for i in range(10)]

    # text = "A elderly lady in a pink top is having a conversation with another elderly lady."
    # input_ids = processor(text=text, return_tensors="pt").input_ids.to(model.device)
    # print(input_ids)

    # embeds = model.language_model.get_input_embeddings()(input_ids)
    # # energy = speaker.energy(embeds, images, 5)
    # grad_energy = torch.func.grad(speaker.energy)(embeds, images, 5)

    # embedding_matrix = (
    #     model.language_model.get_input_embeddings().weight.unsqueeze(0).unsqueeze(0)
    # )

    # S = embeds.size(1)
    # d = embedding_matrix.expand((-1, S, -1, -1)) - embeds.unsqueeze(2)

    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    from speaker import GPT2Speaker

    model = GPT2LMHeadModel.from_pretrained("gpt2", device_map="cuda:0")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    speaker = GPT2Speaker(model, tokenizer)

    model.eval()
    outputs = model.generate(
        do_sample=True,
        output_scores=True,
        return_dict_in_generate=True,
        max_new_tokens=32,
        output_attentions=True,
    )

    print(outputs.sequences)
    scores = torch.gather(
        torch.stack(outputs.scores, dim=1).log_softmax(dim=-1),
        2,
        outputs.sequences[:, 1:].unsqueeze(2),
    )

    print(scores.squeeze())
    print(scores.sum())

    print(
        speaker.energy(
            model.get_input_embeddings()(outputs.sequences[:, 1:]),
            torch.stack(outputs.scores, dim=1),
            outputs.attentions,
        )
    )
