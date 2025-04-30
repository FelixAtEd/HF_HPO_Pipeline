#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import yaml
import argparse
import sys
from dataclasses import dataclass, field
from transformers import HfArgumentParser, TrainingArguments
from typing import Optional


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "Name of a dataset from the datasets package"}
    )
    dataset_path: Optional[str] = field(
        default=None, metadata={"help": "Path to dataset"}
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    validation_fold: Optional[str] = field(
        default=None,
        metadata={"help": "The fold to use as the validation dataset"},
    )
    test_fold: Optional[str] = field(
        default=None,
        metadata={"help": "The fold to use as the test dataset"},
    )
    audio_column_name: str = field(
        default="audio",
        metadata={
            "help": "The name of the dataset column containing the audio data. Defaults to 'audio'"
        },
    )
    label_column_name: str = field(
        default="label",
        metadata={
            "help": "The name of the dataset column containing the labels. Defaults to 'label'"
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_length_seconds: float = field(
        default=20,
        metadata={
            "help": "Audio clips will be randomly cut to this length during training if the value is set."
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="facebook/wav2vec2-base",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from the Hub"
        },
    )
    task: Optional[str] = field(
        default="audio-classification",
        metadata={"help": "Task type"},
    )
    feature_extractor_name: Optional[str] = field(
        default=None, metadata={"help": "Name or path of preprocessor config."}
    )
    freeze_feature_encoder: bool = field(
        default=True,
        metadata={"help": "Whether to freeze the feature encoder layers of the model."},
    )
    sampling_rate: Optional[int] = field(
        default=16000,
        metadata={
            "help": "Sampling rate of the audio.",
        },
    )
    attention_mask: bool = field(
        default=True,
        metadata={
            "help": "Whether to generate an attention mask in the feature extractor."
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={
            "help": "Will enable to load a pretrained model whose head dimensions are different."
        },
    )
    weighted_cross_entropy: bool = field(
        default=False,
        metadata={"help": "Use weighted loss."},
    )
    retrain: bool = field(
        default=True,
        metadata={"help": "Retrain after HPO."},
    )
    task: Optional[str] = field(
        default="audio-classification",
        metadata={"help": "Type of task to run(Audio/Vision)."},
    )


@dataclass
class HPOArguments:
    """Simplified HPO arguments with flat structure"""

    backend: str = field(default="optuna", metadata={"help": "optuna|ray|sigopt"})
    direction: str = field(default="maximize", metadata={"help": "maximize|minimize"})
    n_trials: int = field(default=50, metadata={"help": "Number of trials"})
    # Learning rate range
    lr_min: float = field(
        default=1e-8, metadata={"help": "Min learning rate (log scale)"}
    )
    lr_max: float = field(
        default=1e-3, metadata={"help": "Max learning rate (log scale)"}
    )


def load_yaml_config(config_path: str) -> dict:
    """Load and parse YAML configuration file"""
    with open(config_path) as f:
        return yaml.safe_load(f)


def config_to_args(config: dict) -> list:
    """Convert YAML config dictionary to command-line arguments"""
    args = []
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                args.append(f"--{key}")
        elif isinstance(value, dict):
            for subkey, subvalue in value.items():
                args.extend([f"--{subkey}", str(subvalue)])
        else:
            args.extend([f"--{key}", str(value)])
    return args


def parse_arguments():
    # First parse to get config file location
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--config", type=str)
    base_args, _ = base_parser.parse_known_args()

    # Load YAML config
    yaml_config = load_yaml_config(base_args.config)
    yaml_args = config_to_args(yaml_config)

    # Full argument parser with HF integration
    parser = HfArgumentParser(
        (TrainingArguments, ModelArguments, DataArguments, HPOArguments)
    )
    return parser.parse_args_into_dataclasses(args=yaml_args + sys.argv[3:])


def main():
    # Parse all arguments
    training_args, model_args, data_args, hpo_args = parse_arguments()


if __name__ == "__main__":
    main()
