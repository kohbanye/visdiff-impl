import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from transformers import CLIPModel, CLIPProcessor


class Ranker:
    def __init__(self, clip_model_name="openai/clip-vit-large-patch14"):
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.model = (
            CLIPModel.from_pretrained(clip_model_name).to("cuda")
            if torch.cuda.is_available()
            else CLIPModel.from_pretrained(clip_model_name)
        )

    def _compute_similarity(self, images: torch.Tensor, texts: list[str]):
        inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77,
            do_rescale=False,
        ).to(self.model.device)
        outputs = self.model(**inputs)
        logits_per_image = (
            outputs.logits_per_image
        )  # this is the image-text similarity score
        probs = logits_per_image.softmax(
            dim=1
        )  # we can take the softmax to get probabilities
        return (
            probs[:, 0].detach().cpu().numpy()
        )  # Assuming text at index 0 is the positive text.

    def rank(
        self,
        image_set_a: torch.Tensor,
        image_set_b: torch.Tensor,
        proposed_differences: list[str],
    ) -> list[tuple[str, float]]:
        ranked_differences = []
        for difference in proposed_differences:
            # Correctly compute similarities for each set
            similarities_a = self._compute_similarity(image_set_a, [difference])
            similarities_b = self._compute_similarity(image_set_b, [difference])

            # Efficient AUROC calculation with correct handling of ties
            labels = np.concatenate(
                [np.ones_like(similarities_a), np.zeros_like(similarities_b)]
            )
            scores = np.concatenate([similarities_a, similarities_b])
            auroc = roc_auc_score(labels, scores)

            ranked_differences.append((difference, auroc))

        # Sort by AUROC in descending order (higher AUROC is better)
        ranked_differences.sort(key=lambda x: x[1], reverse=True)
        return ranked_differences
