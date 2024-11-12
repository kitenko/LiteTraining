import pandas as pd


def run_predict_to_csv(predictions: list, class_labels: list) -> None:
    """
    Run prediction process, load sample submission, fill predictions, and save to CSV.
    """

    predicted_indices = [pred["preds"].item() for pred in predictions]
    predicted_labels = [class_labels[idx] for idx in predicted_indices]

    # Load sample submission file
    submission = pd.read_csv("/app/simple_cnn_baseline.csv")
    submission["Expected"] = predicted_labels

    # Save the completed submission
    submission.to_csv("submission.csv", index=False)
    print("Predictions saved to submission.csv")
