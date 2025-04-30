import librosa
import torch
import optuna
import json
from random import randint
import numpy as np
import os
from copy import deepcopy
from scipy.special import softmax
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score,
)

from transformers import (
    TrainingArguments,
    AutoConfig,
    AutoModelForImageClassification,
    AutoModelForAudioClassification,
    AutoFeatureExtractor,
    Trainer,
)
from datasets import DatasetDict, features

from common.custom_metrics import compute_metrics
from common.arguments import (
    ModelArguments,
    HPOArguments,
    DataArguments,
)
from common.training_utilities import (
    PredictionAwareTrainer,
    SaveBestModelCallback,
)


def random_subsample(wav: np.ndarray, max_length: float, sample_rate: int = 16000):
    """Randomly sample chunks of `max_length` seconds from the input audio"""
    sample_length = int(round(sample_rate * max_length))
    if len(wav) <= sample_length:
        return wav
    random_offset = randint(0, len(wav) - sample_length - 1)
    return wav[random_offset : random_offset + sample_length]


# Register any model not implmeneted by HuggingFace
# You will need to implement the following:
# 1. Preprocess function
# 2. HF Config (PretrainedConfig)
# 3. HF Model  (PreTrainedModel)
# 4. Model modality
# {"model_name": {"pre_process" : preprcess_fn,
#            "config":HFConfig,
#            "model": (HFModel, pretrained_model_checkpoint),
#            "modality": "audio|image|text"
#            }
# }

CUSTOM_REGISTRY = {}


class TrainingPipeline:
    # Core registry mapping model prefixes to (processor, model class, modality)
    def __init__(
        self,
        raw_dataset: DatasetDict,
        training_args: TrainingArguments,
        model_args: ModelArguments,
        data_args: DataArguments,
        hpo_args: HPOArguments,
        debug=False,
    ):
        self.debug = debug

        self.custom_model_fns = CUSTOM_REGISTRY.get(model_args.model_name_or_path, None)

        self.training_args = training_args
        self.model_args = model_args
        self.data_args = data_args
        self.hpo_args = hpo_args
        self.labels = (
            raw_dataset["train"].features[self.data_args.label_column_name].names
        )
        self.config = self.get_config()
        self.preprocess_fn = self._load_preprocess_fn()
        self.dataset = self.process_dataset(raw_dataset)
        self.best_model_path = os.path.join(training_args.output_dir, "_best")

    def get_config(self):
        label2id, id2label = {}, {}
        for i, label in enumerate(self.labels):
            label2id[label] = str(i)
            id2label[str(i)] = label
        if self.custom_model_fns:
            config_cls = self.custom_model_fns["config"]
            return config_cls(
                n_class=len(self.labels),
                label2id=label2id,
                id2label=id2label,
                sr=self.model_args.sampling_rate,
            )
        else:
            config_cls = AutoConfig
        model_config = config_cls.from_pretrained(
            self.model_args.model_name_or_path,
            num_labels=len(self.labels),
            label2id=label2id,
            id2label=id2label,
            finetuning_task=self.model_args.task,
            cache_dir=self.model_args.cache_dir,
        )
        return model_config

    def get_model(self, pretrained=True, trained_checkpoint=None):
        if self.custom_model_fns:
            model_cls, pretrained_checkpoint = self.custom_model_fns["model"]
            model = model_cls(self.config)
            if pretrained and not trained_checkpoint:
                return model.from_pretrained(
                    pretrained_model_name_or_path=self.model_args.model_name_or_path,
                    config=self.config,
                )
            elif pretrained and trained_checkpoint:
                return model.from_pretrained(
                    pretrained_model_name_or_path=trained_checkpoint,
                    config=self.config,
                )
            else:
                return model

        elif "audio" in self.model_args.task:
            print("Using Audio Classification Model")
            model_cls = AutoModelForAudioClassification
        elif "image" in self.model_args.task:
            print("Using Image Classification Model")
            model_cls = AutoModelForImageClassification

        if pretrained:
            return model_cls.from_pretrained(
                pretrained_model_name_or_path=(
                    trained_checkpoint
                    if trained_checkpoint is not None
                    else self.model_args.model_name_or_path
                ),
                config=self.config,
                cache_dir=self.model_args.cache_dir,
                ignore_mismatched_sizes=self.model_args.ignore_mismatched_sizes,
            )
        else:
            # If not pretrained, return a new model instance with initialised weights
            return model_cls.from_config(config=self.config)

    def create_output_dir(self, recipe):
        model_name = self.model_args.model_name_or_path

        dataset_name = self.data_args.dataset_name  # Assuming dataset_name is defined

        model_name_nested = model_name.replace("/", "_")

        # Generate a unique output directory based on model name and dataset name
        output_dir = os.path.join(
            self.training_args.output_dir, model_name_nested, dataset_name, recipe
        )

        # Create the output directory (including intermediate directories) if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Optionally, log or print the directory path for debugging
        print(f"Created output directory: {output_dir}")

        self.training_args.output_dir = output_dir

    def create_trainer(self, model_init, loss_fn=None):
        if self.training_args.logging_dir is None:
            self.training_args.logging_dir = f"{self.training_args.output_dir}/logs"
        self.trainer = PredictionAwareTrainer(
            model=None,
            args=self.training_args,
            model_init=model_init,
            train_dataset=(
                self.dataset["train"] if self.training_args.do_train else None
            ),
            eval_dataset=(
                self.dataset["validation"] if self.training_args.do_eval else None
            ),
            compute_metrics=compute_metrics,
            # compute_loss_func=loss_fn or weighted_cross_entropy(self.labels),
            callbacks=[
                SaveBestModelCallback(
                    best_output_dir=f"{self.training_args.output_dir}/_best",
                    metric_for_best_model=self.training_args.metric_for_best_model,
                )
            ],
        )

    def run_trainer_HPO(self):

        def hp_space(trial: optuna.Trial):
            return {
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    float(self.hpo_args.lr_min),
                    float(self.hpo_args.lr_max),
                    log=True,
                ),
                # Matches Trainer class default creation of trial folders for good tensorboard logging
                "logging_dir": f"{self.training_args.output_dir}/run-{trial.number}/logs",
            }

        def compute_objective(metrics):
            return metrics[self.training_args.metric_for_best_model]

        best_trial = self.trainer.hyperparameter_search(
            direction=self.hpo_args.direction,
            backend=self.hpo_args.backend,
            hp_space=hp_space,
            n_trials=self.hpo_args.n_trials,
            compute_objective=compute_objective,
            # Huggingface's optuna integration does not implment resuming from study
            # The study used a modified version of transformer package to allow resuming from a study
            study_name=f"study",
            storage=f"sqlite:///{self.training_args.output_dir}/study.db",
            load_if_exists=True,
        )
        print(
            f"Using load_best_model_at_end: {self.trainer.args.load_best_model_at_end}"
        )
        print(f"Using metric_for_best_model: {self.trainer.args.metric_for_best_model}")

        return best_trial

    def process_dataset(self, dataset: DatasetDict):

        if self.training_args.do_train:
            if self.data_args.max_train_samples is not None:
                dataset["train"] = (
                    dataset["train"]
                    .shuffle(seed=self.training_args.seed)
                    .select(range(self.data_args.max_train_samples))
                )
        if self.training_args.do_eval:
            if self.data_args.max_eval_samples is not None:
                dataset["validation"] = (
                    dataset["validation"]
                    .shuffle(seed=self.training_args.seed)
                    .select(range(self.data_args.max_eval_samples))
                )

        dataset = dataset.cast_column(
            self.data_args.audio_column_name,
            features.Audio(sampling_rate=self.model_args.sampling_rate),
        )

        dataset_cache_dir = f"./dataset_cache/{self.model_args.model_name_or_path.replace('/','_')}/{self.data_args.dataset_name}"

        # Ensure the directory exists
        os.makedirs(dataset_cache_dir, exist_ok=True)

        if self.training_args.eval_strategy != "no":
            dataset["train"] = dataset["train"].map(
                self.preprocess_fn[0],
                batched=True,
                num_proc=1,
                cache_file_name=(
                    f"{dataset_cache_dir}/train_cache.arrow" if not self.debug else None
                ),
            )

            dataset["validation"] = dataset["validation"].map(
                self.preprocess_fn[1],
                batched=True,
                num_proc=1,
                cache_file_name=(
                    f"{dataset_cache_dir}/val_cache.arrow" if not self.debug else None
                ),
            )
        if "test" in dataset:
            dataset["test"] = dataset["test"].map(
                self.preprocess_fn[1],
                batched=True,
                num_proc=1,
                cache_file_name=(
                    f"{dataset_cache_dir}/test_cache.arrow" if not self.debug else None
                ),
            )
        print(dataset)
        return dataset

    def _load_preprocess_fn(self):
        if self.custom_model_fns:
            _preprocess_fn = self.custom_model_fns["pre_process"](self)
            return _preprocess_fn
        elif "audio" in self.model_args.task:
            return self._preprocess_audio()
        elif "image" in self.model_args.task:
            return self._preprocess_vision()

    def _preprocess_audio(self):
        # ASTFeatureExtractor -> Extracts audio features to Mel spectrogram with an input size of [1024, 128] to match pretrained model
        # Wav2Vec2FeatureExtractor -> Only performs normalization if do_normalize is set to True
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            self.model_args.feature_extractor_name
            or self.model_args.model_name_or_path,
            return_attention_mask=self.model_args.attention_mask,
            cache_dir=self.model_args.cache_dir,
        )

        model_input_name = feature_extractor.model_input_names[0]

        def train_transforms(batch):
            """Apply train_transforms across a batch."""
            subsampled_wavs = []
            # Randomly subsample the audio to max_length_seconds
            # For reproducability, all inputs are pre-process to ensure they are less than the max_length
            for audio in batch[self.data_args.audio_column_name]:
                wav = random_subsample(
                    audio["array"],
                    max_length=self.data_args.max_length_seconds,
                    sample_rate=feature_extractor.sampling_rate,
                )
                subsampled_wavs.append(wav)
            inputs = feature_extractor(
                subsampled_wavs, sampling_rate=feature_extractor.sampling_rate
            )
            output_batch = {model_input_name: inputs.get(model_input_name)}
            output_batch["labels"] = list(batch[self.data_args.label_column_name])

            return output_batch

        def val_transforms(batch):
            """Apply val_transforms across a batch."""
            wavs = [audio["array"] for audio in batch[self.data_args.audio_column_name]]
            inputs = feature_extractor(
                wavs, sampling_rate=feature_extractor.sampling_rate
            )
            output_batch = {model_input_name: inputs.get(model_input_name)}
            output_batch["labels"] = list(batch[self.data_args.label_column_name])

            return output_batch

        return train_transforms, val_transforms

    def _preprocess_vision(self):
        def mfcc_transforms(batch):
            audio_arrays = [x["array"] for x in batch["input_values"]]
            sr_list = [x["sampling_rate"] for x in batch["input_values"]]
            mfcc_list = []
            for audio, sr in zip(audio_arrays, sr_list):
                if self.model_args.sampling_rate == 16000:
                    mfcc = librosa.feature.mfcc(
                        y=audio,
                        sr=sr,
                        n_mfcc=16,
                        n_fft=512,
                        hop_length=128,
                        win_length=512,
                        fmin=0.0,
                        n_mels=32,
                        fmax=self.model_args.sampling_rate / 2,
                    )
                else:
                    mfcc = librosa.feature.mfcc(
                        y=audio,
                        sr=sr,
                        n_mfcc=64,
                        n_fft=1024,
                        hop_length=512,
                        win_length=1024,
                        fmin=0.0,
                        fmax=self.model_args.sampling_rate / 2,
                    )
                # Take the mean of the MFCC features along the time axis
                mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
                # We resize for data2vec-vision as it requires an input size of 224x224
                # Other models handles the resizing internally
                if (
                    "facebook/data2vec-vision-base"
                    in self.model_args.model_name_or_path
                ):
                    mfcc = F.interpolate(
                        torch.tensor(mfcc).unsqueeze(0).unsqueeze(0),
                        size=(224, 224),
                        mode="bilinear",
                        align_corners=False,
                    )
                    mfcc = mfcc.squeeze(0).squeeze(0)
                mfcc_list.append(mfcc)

            batch["pixel_values"] = [
                torch.tensor(x).unsqueeze(0).repeat(3, 1, 1) for x in mfcc_list
            ]
            return batch

        return mfcc_transforms, mfcc_transforms

    def test_model(self, test_model, best_model_path):
        test_args = deepcopy(self.training_args)
        test_args.eval_strategy = "no"
        test_args.do_train = False
        test_args.do_eval = False
        test_args.do_predict = True
        test_trainer = Trainer(
            model=test_model,
            args=test_args,
            compute_metrics=compute_metrics,
        )
        results = test_trainer.predict(self.dataset["test"])
        print(results.metrics)
        self.create_model_card(
            results, best_model_path, test_trainer.model.config.id2label
        )
        return test_trainer, results

    def create_model_card(self, test_results, best_model_path, id2label):
        def format_metric(value):
            try:
                return f"{float(value):.4f}"
            except (TypeError, ValueError):
                return str(value)

        # Infer architecture from model name
        model_name = self.model_args.model_name_or_path
        architecture_map = {
            "data2vec-audio": "Data2VecAudio",
            "ast": "AudioSpectrogramTransformer",
            "ssast": "SSAST",
            "wav2vec": "Wav2Vec2",
            "resnet-18": "ResNet18",
            "resnet-50": "ResNet50",
        }
        architecture = next(
            (v for k, v in architecture_map.items() if k in model_name.lower()),
            "CustomAudioModel",
        )

        # Generate metrics
        first_row = test_results.label_ids[0]
        if (
            first_row.ndim == 1
            and np.all((first_row == 0) | (first_row == 1))
            and np.sum(first_row) == 1
        ):
            print("One-hot encoded labels detected. Converting to label indices.")
            y_true = np.argmax(test_results.label_ids, axis=1)
        else:
            print("Label indices detected.")
            y_true = test_results.label_ids

        if np.allclose(test_results.predictions.sum(axis=1), 1, atol=1e-6):
            print("Model outputs probabilities. Using as is.")
            y_probs = test_results.predictions  # Already probabilities
            y_pred = np.argmax(test_results.predictions, axis=1)
        else:
            print("Model outputs logits. Applying softmax.")
            y_probs = softmax(test_results.predictions, axis=-1)
            y_pred = np.argmax(test_results.predictions, axis=1)

        report = classification_report(y_true, y_pred, output_dict=True)
        mcc = matthews_corrcoef(y_true, y_pred)
        auc = roc_auc_score(y_true, y_probs, multi_class="ovr")

        # Confusion matrix
        class_names = [id2label[i] for i in sorted(id2label.keys())]

        # Generate confusion matrix with labels
        cm = confusion_matrix(y_true, y_pred, normalize="true")
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        cm_path = os.path.join(best_model_path, "confusion_matrix.png")
        plt.savefig(cm_path, bbox_inches="tight")
        plt.close()
        val_fold_info = ""
        if (
            hasattr(self.data_args, "validation_fold")
            and self.data_args.validation_fold is not None
        ):
            val_fold_info = f" (Fold {self.data_args.validation_fold})"

        test_fold_info = ""
        if (
            hasattr(self.data_args, "test_fold")
            and self.data_args.test_fold is not None
        ):
            test_fold_info = f" (Fold {self.data_args.test_fold})"
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )

        with open(os.path.join(best_model_path, "trainer_state.json"), "r") as file:
            training_data = json.load(file)

        best_metric = training_data["best_metric"]
        best_validation_metrics = None
        for log in training_data["log_history"]:
            if "eval_recall" in log and log["eval_recall"] == best_metric:
                best_validation_metrics = log
                break

        batch_size = training_data.get("train_batch_size", "N/A")
        if world_size > 1:
            batch_size *= world_size

        model_card = f"""
## Model Details

- **Model Name**: `{os.path.basename(model_name)}`
- **Architecture**: {architecture}
- **Base Model**: [{model_name}]
- **Task**: {self.model_args.task.title()}
- **Sampling Rate**: {self.model_args.sampling_rate} Hz

## Dataset

- **Name**: {self.data_args.dataset_name}
- **Splits**:
- Training Samples: {len(self.dataset['train'])}
- Validation{val_fold_info} Samples: {len(self.dataset['validation'])}
- Test{test_fold_info} Samples: {len(self.dataset['test'])}


## Training Configuration

### Hyperparameters
| Parameter                   | Value       |
|-----------------------------|-------------|
| Best Model Checkpoint       | {os.path.relpath(training_data['best_model_checkpoint'], os.getcwd())} |
| Learning Rate               | {training_data['trial_params']['learning_rate']} |
| Batch Size (Train/Eval)     | {batch_size} |
| Training Steps              | {training_data.get('max_steps', 'N/A')} |

## Validation Metrics

| Metric       | Value       |
|--------------|-------------|
| Accuracy     | {format_metric(best_validation_metrics.get('eval_accuracy', 'N/A'))} |
| Recall (Macro) | {format_metric(best_validation_metrics.get('eval_recall', 'N/A'))} |
| Loss          | {format_metric(best_validation_metrics.get('eval_loss', 'N/A'))} |
| ROC AUC       | {format_metric(best_validation_metrics.get('eval_roc_auc', 'N/A'))} |

## Testing Results

| Metric       | Value       |
|--------------|-------------|
| Accuracy     | {test_results.metrics['test_accuracy']:.4f} |
| Recall (Macro) | {test_results.metrics['test_recall']:.4f} |
| Precision (Macro) | {report['macro avg']['precision']:.4f} |
| F1 (Macro)   | {report['macro avg']['f1-score']:.4f} |
| MCC          | {mcc:.4f} |
| ROC AUC      | {auc:.4f} |

![Confusion Matrix](confusion_matrix.png)

        """
        card_path = os.path.join(best_model_path, "README.md")
        with open(card_path, "w") as f:
            f.write(model_card)
