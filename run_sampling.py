from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)
from decoding.speaker import Blip2Speaker
from decoding.listener import Blip2Listener
from decoding.langevin import pNCG
import torch
from PIL import Image
from pathlib import Path
import json
from copy import deepcopy


def main(data_file, save_file, n_captions):
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b-coco")

    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b-coco",
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
    )

    speaker = Blip2Speaker(model, processor)
    listener = Blip2Listener(model, processor)

    with open(data_file) as f:
        data = json.load(f)

    image_path = Path(
        "/data/tir/projects/tir3/users/svadugur/pragmatic-clip/image-sets"
    )
    for x in data:
        images = [
            Image.open(image_path / x["image_set"] / f"img{i}.jpg") for i in range(10)
        ]
        inputs = processor(
            images=images[int(x["image_index"])], return_tensors="pt"
        ).to(model.device)

        captions = []
        for i in range(n_captions):
            outputs = model.generate(
                **inputs,
                do_sample=True,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=32,
            )
            init_state = outputs.sequences[0].unsqueeze(0)
            cutoff = (
                (outputs.sequences[0] == 50140).nonzero()[0]
                if torch.any(outputs.sequences[0] == 50140)
                else outputs.sequences[0].shape[0]
            )
            init_state = init_state[:, :cutoff]

            try:
                sample = pNCG(
                    images,
                    int(x["image_index"]),
                    init_state.shape[1],
                    speaker.energy,
                    listener.energy,
                    model.language_model.get_input_embeddings(),
                    100,
                    5.0,
                    10,
                    10,
                    0.5,
                    processor.tokenizer,
                    model.device,
                    init_state=model.language_model.get_input_embeddings()(init_state),
                )
                captions.append(
                    speaker.processor.batch_decode(sample, skip_special_tokens=True)
                )
            except:
                continue
        x["captions"] = captions

        with open(save_file, "w") as f:
            json.dump(data, f)


if __name__ == "__main__":
    import sys

    data_file = sys.argv[1]
    save_file = sys.argv[2]
    n_captions = int(sys.argv[3])

    main(data_file, save_file, n_captions)
