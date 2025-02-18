import logging

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import CLIPModel, CLIPProcessor


class Ranker:
    def __init__(self, clip_model_name="openai/clip-vit-large-patch14"):
        self.logger = logging.getLogger(__name__)
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
        probs = outputs.logits_per_image.softmax(dim=0)
        return probs[:, 0].detach().cpu().numpy()

    def rank(
        self,
        image_set_a: torch.Tensor,
        image_set_b: torch.Tensor,
        proposed_differences: list[str],
    ) -> list[tuple[str, float]]:
        ranked_differences = []
        self.logger.info("Ranking differences...")
        for difference in proposed_differences:
            similarities_a = self._compute_similarity(image_set_a, [difference])
            similarities_b = self._compute_similarity(image_set_b, [difference])

            labels = np.concatenate(
                [np.ones_like(similarities_a), np.zeros_like(similarities_b)]
            )
            scores = np.concatenate([similarities_a, similarities_b])
            auroc = roc_auc_score(labels, scores)

            ranked_differences.append((difference, auroc))

        # Sort by AUROC in descending order
        ranked_differences.sort(key=lambda x: x[1], reverse=True)
        return ranked_differences
