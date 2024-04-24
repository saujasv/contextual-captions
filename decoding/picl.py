from transformers import LogitsProcessor, ProcessorMixin
from typing import List, Union, Optional
from pathlib import Path
import open_clip
import torchvision.transforms.transforms as T
from listener import CLIPListener, Listener
from PIL import Image
import ipdb


class PICLLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        listener: Listener,
        contexts: List[List[Union[Path, str, Image.Image]]],
        targets: List[int],
        processor: ProcessorMixin,
        informativity: float,
        adaptive_plausibility_alpha: float = 0.1,
        beam_size: int = 100,
    ):
        super().__init__()

        self.listener = listener
        self.contexts = contexts
        try:
            self.context_features = listener.encode_images(contexts)
        except NotImplementedError:
            self.context_features = None
        self.targets = targets
        self.processor = processor
        self.informativity = informativity
        self.adaptive_plausibility_alpha = adaptive_plausibility_alpha
        self.beam_size = beam_size
        self.prefix_length = None

    def __call__(self, input_ids, scores):
        if self.prefix_length is None:
            self.prefix_length = input_ids.size(1)

        log_p = scores.log_softmax(dim=-1)

        V_head = torch.ge(
            log_p,
            (torch.tensor(self.adaptive_plausibility_alpha).log() + log_p)
            .max(axis=1)
            .values.unsqueeze(1),
        )

        partial_tokens = torch.cat((input_ids[torch.where(V_head)[0]], torch.where(V_head)[1].unsqueeze(1)), dim=1)
        partial_texts = self.processor.batch_decode(partial_tokens[:, self.prefix_length:], skip_special_tokens=True)

        listener_scores = self.listener.score_texts(
            partial_texts, image_features=self.context_features
        )

        log_p[*torch.where(V_head)] = (1 - self.informativity) * log_p[*torch.where(V_head)] + self.informativity * listener_scores[0, self.targets, :].to(log_p.device, log_p.dtype)

        log_p.masked_fill_(~V_head, float("-inf"))

        return log_p

if __name__ == "__main__":
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    import torch
    from PIL import Image
    import requests
    import open_clip
    from pathlib import Path
    import os
    import ipdb

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b-coco")

    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b-coco",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )

    # prepare image and text prompt, using the appropriate prompt template
    url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
    image = Image.open(requests.get(url, stream=True).raw)
    prompt = ""

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        model_name="ViT-B-32-quickgelu", pretrained="metaclip_fullcc", device="cuda:1", precision="bfloat16"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")
    listener = CLIPListener(clip_model, clip_preprocess, tokenizer, "cuda:1")

    image_path = Path(
        "/data/tir/projects/tir3/users/svadugur/pragmatic-clip/image-sets"
    )
    image_sets = [
        "open-images-2057_2fc6afbbb663b164",
        "open-images-2057_2fc6afbbb663b164",
        "open-images-2057_2fc6afbbb663b164",
        "open-images-74_13ecb782007c9c85",
    ]
    contexts = [
        [Image.open(image_path / ctx / f"img{i}.jpg") for i in range(10)]
        for ctx in image_sets
    ]
    targets = list(map(int, ["5", "7", "2", "2"]))

    inputs = processor(
        # text=[prompt for i in range(len(contexts))][:1],
        images=[ctx[t] for ctx, t in zip(contexts, targets)][:1],
        return_tensors="pt",
    ).to("cuda:0")

    # autoregressively complete prompt
    output = model.generate(
        **inputs,
        max_new_tokens=64,
        logits_processor=[
            PICLLogitsProcessor(
                listener,
                contexts[:1],
                targets[:1],
                processor,
                0.9,
                0.01
            )
        ],
        do_sample=True,
        num_return_sequences=8,
        num_beams=1,
    )

    print(*[cap.strip() for cap in processor.batch_decode(output, skip_special_tokens=True)], sep='\nNEXT\n')
