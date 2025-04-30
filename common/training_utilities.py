import os
import shutil
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from transformers import Trainer, TrainerCallback


class PredictionAwareTrainer(Trainer):
    def evaluation_loop(self, *args, **kwargs):
        """Subclass the evaluation loop to generate extra metrics."""
        # Run standard evaluation
        output = super().evaluation_loop(*args, **kwargs)

        # Generate artifacts using only local variables
        self._generate_artifacts(
            predictions=output.predictions,
            label_ids=output.label_ids,
        )

        return output

    def _generate_artifacts(self, predictions: np.ndarray, label_ids: np.ndarray):
        # Create output directory
        if not hasattr(self, "_trial") or self._trial is None:
            pred_output_dir = Path(f"{self.args.output_dir}/eval_predictions")
        else:
            pred_output_dir = Path(
                f"{self.args.output_dir}/run-{self._trial.number}/eval_predictions"
            )

        pred_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            if isinstance(predictions, tuple):
                predictions = predictions[1]
            preds = np.argmax(predictions, axis=1)
            if label_ids.ndim == 2 and label_ids.shape[1] > 1:
                label_ids = np.argmax(
                    label_ids, axis=1
                )  # Convert one-hot to single label

            cm = confusion_matrix(label_ids, preds)

            id2label = self.model.config.id2label
            labels = list(
                id2label.values()
            )  # Extract labels (keys of id2label are IDs)
            # Save raw predictions
            np.savez_compressed(
                f"{pred_output_dir}/step_{self.state.global_step}.npz",
                predictions=predictions,
                label_ids=label_ids,
                id2label=id2label,
            )
            cm_percent = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
            # Create the confusion matrix heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                cm_percent,
                annot=True,
                fmt=".2f",
                cmap="Blues",
                ax=ax,
                xticklabels=labels,
                yticklabels=labels,
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title("Confusion Matrix")

            # Get the logging directory from Trainer arguments
            writer = SummaryWriter(log_dir=self.args.logging_dir)

            # Log the confusion matrix figure to TensorBoard
            writer.add_figure(
                "eval/confusion_matrix", fig, global_step=self.state.global_step
            )
            writer.close()
        except Exception as e:
            print(f"Artifact generation failed: {str(e)}")
            raise
        finally:
            # Explicit cleanup (not strictly needed, but good practice)
            del predictions
            del label_ids


class SaveBestModelCallback(TrainerCallback):
    def __init__(
        self,
        best_output_dir="output/best",
        metric_for_best_model="eval_recall",
        greater_is_better=True,
    ):
        self.best_output_dir = best_output_dir
        self.metric_for_best_model = metric_for_best_model
        self.greater_is_better = greater_is_better
        self.best_metric = None

    def on_save(self, args, state, control, **kwargs):
        """
        Hook called when the model is saved during training or hyperparameter search.
        """
        # Get the current metric value from the state
        if not state.best_metric:
            return

        metric_value = state.best_metric

        # Check if this is the best model so far
        is_best = self.best_metric is None or (
            metric_value > self.best_metric
            if self.greater_is_better
            else metric_value < self.best_metric
        )
        if is_best:
            print("=" * 10)
            print(
                f"Saving best model with {self.metric_for_best_model}:{state.best_metric} with trial_params:{state.trial_params}"
            )
            print("=" * 10)

            self.best_metric = metric_value

            # Save only the checkpoint files
            os.makedirs(self.best_output_dir, exist_ok=True)
            shutil.copytree(
                state.best_model_checkpoint, self.best_output_dir, dirs_exist_ok=True
            )


def weighted_cross_entropy(train_labels):
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(train_labels), y=train_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    def compute_loss_func(outputs, labels, num_items_in_batch=None):
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=class_weights.to(outputs.logits.device)
        )
        return loss_fct(outputs.logits, labels)

    return compute_loss_func
