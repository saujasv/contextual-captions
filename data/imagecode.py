from pathlib import Path
from datasets import load_dataset
from typing import List, Union, Optional
from torch.utils.data import DataLoader


class ImageCoDeCollator:
    """
    Collator for the ImageCoDe dataset. This collator takes a batch of samples
    and creates a dictionary with the following keys:
    - image_set: A list of contexts for each instance. Each context is a
        list of *absolute* paths to the image files
    - image_index: A list of integers, each of which is the index of the target
        image in the corresponding element of `image_set`
    - description: A list of descriptions of the target image in each context 
        in `image_set`
    """

    def __init__(self, base_path: Union[str, Path]):
        self.base_path = base_path if isinstance(base_path, Path) else Path(base_path)

    def __call__(self, batch):
        img_idx = [int(x["image_index"]) for x in batch]
        img_files = [
            list((self.base_path / x["image_set"]).glob("*.jpg")) for x in batch
        ]
        img_files = [
            sorted(x, key=lambda p: int(str(p).split("/")[-1].split(".")[0][3:]))
            for x in img_files
        ]
        text = [x["description"] for x in batch]
        batch = {"image_index": img_idx, "image_set": img_files, "description": text}
        return batch


def get_imagecode_dataloader(
    image_sets_path: Union[str, Path],
    batch_size: int,
    splits: Optional[List[str]] = ["train", "validation"],
    static_only: bool = True,
    shuffle: bool = False,
):
    dataset = load_dataset("BennoKrojer/ImageCoDe", split=splits)
    if static_only:
        # All the static images are from the OpenImages dataset sop we can
        # use the image set identifier to filter out the the images from videos
        dataset = dataset.filter(lambda x: "open-images" in x["image_set"])

    collator = ImageCoDeCollator(image_sets_path)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
    )
