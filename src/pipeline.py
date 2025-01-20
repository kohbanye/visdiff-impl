from data import VisDiffDataset, VisDiffDatasetName
from proposer import Proposer
from ranker import Ranker


class VisDiffPipeline:
    def __init__(self, proposer: Proposer, ranker: Ranker):
        self.proposer = proposer
        self.ranker = ranker

    def predict(self, dataset: VisDiffDataset, num_predictions: int = 5):
        results = []
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
                {
                    "set1_label": data.set1_label,
                    "set2_label": data.set2_label,
                    "ground_truth_difference": data.difference,
                    "predicted_differences": top_predictions,
                }
            )

        return results


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    dataset = VisDiffDataset(name=VisDiffDatasetName.EASY)

    proposer = Proposer()
    ranker = Ranker()

    pipeline = VisDiffPipeline(proposer, ranker)

    predictions = pipeline.predict(dataset)

    for prediction in predictions:
        print(f"Set 1: {prediction['set1_label']}, Set 2: {prediction['set2_label']}")
        print(f"Ground Truth Difference: {prediction['ground_truth_difference']}")
        print("Predicted Differences:")
        for diff in prediction["predicted_differences"]:
            print(f"- {diff}")
        print("-" * 20)
