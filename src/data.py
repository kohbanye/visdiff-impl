import os
from enum import Enum

import torch
import torchvision
from PIL import Image
from pydantic import BaseModel
from torch.utils.data import Dataset

DATASET_DIR = "VisDiffBench-release"


class VisDiffDatasetName(Enum):
    EASY = "easy"
    EASY_REDUCED = "easy_reduced"
    MEDIUM = "medium"
    HARD = "hard"
    HARD_REDUCED = "hard_reduced"
    IMAGENETR = "imagenetr"
    IMAGENETSTAR = "imagenetstar"
    MINE = "mine"


class ImageSetInfo(BaseModel):
    set1: str
    set2: str
    difference: str
    set1_images: list[str]
    set2_images: list[str]
    set1_images_url: list[str] | None = None
    set2_images_url: list[str] | None = None


class VisDiffData(BaseModel, arbitrary_types_allowed=True):
    set1_images: list[Image.Image]
    set2_images: list[Image.Image]
    set1_images_tensor: torch.Tensor
    set2_images_tensor: torch.Tensor
    set1_label: str
    set2_label: str
    difference: str


class VisDiffDataset(Dataset):
    def __init__(self, name: VisDiffDatasetName):
        self.name = name
        self.jsonl_path = os.path.join(DATASET_DIR, f"{name.value}.jsonl")
        with open(self.jsonl_path, "r") as f:
            self.json_list = f.read().splitlines()

        self.image_set_info_list: list[ImageSetInfo] = []
        for json_str in self.json_list:
            image_set_info = ImageSetInfo.model_validate_json(json_str)
            self.image_set_info_list.append(image_set_info)

    def __len__(self):
        return len(self.image_set_info_list)

    def __getitem__(self, idx: int) -> VisDiffData:
        image_set_info = self.image_set_info_list[idx]
        set1_image_paths = image_set_info.set1_images
        set1_images = [
            Image.open(f"{DATASET_DIR}/{path}").convert("RGB")
            for path in set1_image_paths
        ]
        set2_image_paths = image_set_info.set2_images
        set2_images = [
            Image.open(f"{DATASET_DIR}/{path}").convert("RGB")
            for path in set2_image_paths
        ]

        tensor_transform = torchvision.transforms.ToTensor()
        resize_transform = torchvision.transforms.Resize((512, 512))

        set1_images_resized = [resize_transform(img) for img in set1_images]
        set2_images_resized = [resize_transform(img) for img in set2_images]

        set1_images_tensor = torch.stack(
            [tensor_transform(img) for img in set1_images_resized]
        )
        set2_images_tensor = torch.stack(
            [tensor_transform(img) for img in set2_images_resized]
        )

        return VisDiffData(
            set1_images=set1_images,
            set2_images=set2_images,
            set1_images_tensor=set1_images_tensor,
            set2_images_tensor=set2_images_tensor,
            set1_label=image_set_info.set1,
            set2_label=image_set_info.set2,
            difference=image_set_info.difference,
        )
