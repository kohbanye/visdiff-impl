import logging

from openai import OpenAI
from pydantic import BaseModel


class Prediction(BaseModel):
    set1_label: str
    set2_label: str
    ground_truth_difference: str
    predicted_differences: list[str]


class Metrics(BaseModel):
    accuracy: float
    accuracy_top_1: float
    accuracy_top_5: float


class Evaluator:
    def __init__(self, llm_model_name="gpt-4o"):
        self.logger = logging.getLogger(__name__)
        self.openai_client = OpenAI()
        self.llm_model_name = llm_model_name

    def _get_prompt(
        self, set1_label: str, set2_label: str, predicted_difference: str
    ) -> str:
        return f"""\
You are trying to summarize differences in groups of images. The goal is to find a concept that is more true for Group A than Group B.
Given a description of Group A and Group B, output whether a given prediction aligns with the description of Group A. Answer with a 2 (fully aligned), 1 (somewhat aligned), or 0 (not aligned). a score of 1 should be given if the prediction is more true for A than B, but is a superset or a subset of the most correct difference.

For example, if Group A is \"images of dogs in the snow\" and Group B is \"images of dogs next to cats\":
    - predictions like \"dogs in the snow\" or \"dogs in winter time\" should be given a 2
    - predictions like \"golden retrivers on a ski slope\" or \"animals in the snow\" should be given a 1

Here is the descriptions of the groups:
Group A: {set1_label}
Group B: {set2_label}
Prediction: {predicted_difference}

Again, output either a 2, 1, or 0.
"""

    def evaluate(self, prediction: Prediction) -> Metrics:
        scores: list[int] = []

        for predicted_difference in prediction.predicted_differences:
            prompt = self._get_prompt(
                prediction.set1_label, prediction.set2_label, predicted_difference
            )

            self.logger.info("Requesting completion to evaluate prediction...")
            completion = self.openai_client.chat.completions.create(
                model=self.llm_model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a evaluator trying to evaluate a prediction based on descriptions of two groups of images.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            response = completion.choices[0].message.content
            score = 0
            if response is None:
                self.logger.error("Failed to get a response from the model.")
                scores.append(0)
                continue
            try:
                score = int(response)
            except ValueError:
                self.logger.error(
                    f"Failed to parse response from the model: {response}"
                )
                scores.append(0)
                continue

            if score not in [0, 1, 2]:
                self.logger.error(f"Invalid response from the model: {response}")
                scores.append(0)
                continue

            scores.append(score)

        normalized_scores = [score / 2 for score in scores]
        return Metrics(
            accuracy=sum(normalized_scores) / len(normalized_scores),
            accuracy_top_1=normalized_scores[0],
            accuracy_top_5=sum(normalized_scores[:5]) / min(5, len(normalized_scores)),
        )
