from pydantic import BaseModel

from data import VisDiffDataset, VisDiffDatasetName
from evaluate import Evaluator, Metrics, Prediction
from proposer import Proposer
from ranker import Ranker


class Results(BaseModel):
    predictions: list[Prediction]
    metrics: list[Metrics]


class VisDiffPipeline:
    def __init__(self, proposer: Proposer, ranker: Ranker, evaluator: Evaluator):
        self.proposer = proposer
        self.ranker = ranker
        self.evaluator = evaluator
        self.predictions: list[Prediction] = []

    def predict(
        self, dataset: VisDiffDataset, num_predictions: int = 5
    ) -> list[Prediction]:
        results: list[Prediction] = []
        for i in range(len(dataset)):
            data = dataset[i]

            proposed_differences = self.proposer.propose(
                data.set1_images, data.set2_images
            )
            ranked_differences = self.ranker.rank(
                data.set1_images_tensor, data.set2_images_tensor, proposed_differences
            )

            top_predictions = [diff[0] for diff in ranked_differences[:num_predictions]]

            results.append(
                Prediction(
                    set1_label=data.set1_label,
                    set2_label=data.set2_label,
                    ground_truth_difference=data.difference,
                    predicted_differences=top_predictions,
                )
            )

        return results

    def evaluate(self, predictions: list[Prediction]) -> list[Metrics]:
        metrics_list: list[Metrics] = []
        for prediction in predictions:
            metrics = self.evaluator.evaluate(prediction)
            metrics_list.append(metrics)
        return metrics_list


if __name__ == "__main__":
    import argparse
    import logging

    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[name.value for name in VisDiffDatasetName],
        default=VisDiffDatasetName.EASY_REDUCED.value,
    )
    args = parser.parse_args()

    dataset_name = VisDiffDatasetName(args.dataset)
    dataset = VisDiffDataset(name=dataset_name)

    proposer = Proposer(caption_model_name="Salesforce/blip2-opt-6.7b-coco")
    ranker = Ranker()
    evaluator = Evaluator()

    pipeline = VisDiffPipeline(proposer, ranker, evaluator)

    predictions = pipeline.predict(dataset)
    metrics_list = pipeline.evaluate(predictions)

    results = Results(predictions=predictions, metrics=metrics_list)
    with open(f"results_{dataset_name.value}.json", "w") as f:
        f.write(results.model_dump_json(indent=2))

    for prediction, metrics in zip(predictions, metrics_list):
        print(f"Set 1: {prediction.set1_label}, Set 2: {prediction.set2_label}")
        print(f"Ground Truth Difference: {prediction.ground_truth_difference}")
        print("Predicted Differences:")
        for diff in prediction.predicted_differences:
            print(f"- {diff}")
        print(f"Accuracy: {metrics.accuracy}")
        print(f"Accuracy Top 1: {metrics.accuracy_top_1}")
        print(f"Accuracy Top 5: {metrics.accuracy_top_5}")
        print("-" * 20)
