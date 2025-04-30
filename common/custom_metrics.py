import numpy as np
from scipy.special import softmax
import evaluate

cm_metric = evaluate.load("confusion_matrix")
roc_auc_score = evaluate.load("roc_auc", "multiclass")
accuracy = evaluate.load("accuracy")
recall = evaluate.load("recall")


def compute_metrics(eval_pred):

    if np.allclose(eval_pred.predictions.sum(axis=1), 1, atol=1e-6):
        print("Model outputs probabilities. Using as is.")
        prob_predictions = eval_pred.predictions  # Already probabilities
    else:
        print("Model outputs logits. Applying softmax.")
        prob_predictions = softmax(eval_pred.predictions, axis=-1)

    if eval_pred.label_ids.ndim == 2 and np.allclose(
        np.sum(eval_pred.label_ids, axis=1), 1.0, rtol=1e-2, atol=1e-2
    ):
        print("One-hot encoded labels detected. Converting to label indices.")
        references = np.argmax(eval_pred.label_ids, axis=1)
    else:
        print("Label indices detected.")
        references = eval_pred.label_ids

    predictions = np.argmax(eval_pred.predictions, axis=1)

    roc_auc_value = float("nan")
    try:
        unique_labels = np.unique(references)
        if len(unique_labels) >= 2:
            roc_auc_result = roc_auc_score.compute(
                prediction_scores=prob_predictions,
                references=references,
                multi_class="ovr",
            )
            roc_auc_value = roc_auc_result["roc_auc"]
    except ValueError as e:
        print(f"Error computing ROC AUC: {e}")

    metrics = {
        "accuracy": accuracy.compute(predictions=predictions, references=references)[
            "accuracy"
        ],
        "recall": recall.compute(
            predictions=predictions, references=references, average="macro"
        )["recall"],
        "roc_auc": roc_auc_value,
    }
    return metrics
