import os
import yaml
from pathlib import Path
from datasets import load_dataset, ClassLabel


# Load the main dataset configuration (datasets.yaml)
def load_main_dataset_config():
    with open("configs/dataset_registery.yaml", "r") as f:
        return yaml.safe_load(f)


def load_recipe_config(recipe_file):
    with open(recipe_file, "r") as f:
        return yaml.safe_load(f)


def load_from_splits(data_dir, splits_config):

    csv_files = {
        "train": str(data_dir / splits_config["train"]),
        "validation": str(data_dir / splits_config["validation"]),
        "test": str(data_dir / splits_config["test"]),
    }

    # Load the dataset from the CSV files using Hugging Face's load_dataset
    return load_dataset("csv", data_files=csv_files, delimiter="\t")


def load_from_folds(data_dir, dataset_config, dataset_recipe_config):

    dataset_folds_config = dataset_config["folds"]
    valid_fold = dataset_recipe_config.validation_fold
    test_fold = dataset_recipe_config.test_fold

    if not valid_fold:
        raise ValueError("Validation fold must be specified.")

    validation_path = dataset_folds_config.get(f"fold_{valid_fold}")
    if validation_path is None:
        raise ValueError(
            f"Validation fold {valid_fold} not found in dataset_folds_config."
        )

    # Collect remaining folds for training
    training_paths = [
        os.path.join(data_dir, fold_path)
        for fold_name, fold_path in dataset_folds_config.items()
        if fold_name != f"fold_{valid_fold}"
        and (not test_fold or fold_name != f"fold_{test_fold}")
    ]

    csv_files = {
        "train": training_paths,
        "validation": os.path.join(data_dir, validation_path),
    }

    if test_fold:
        test_path = dataset_folds_config.get(f"fold_{test_fold}")
        if test_path:
            csv_files["test"] = os.path.join(data_dir, test_path)

    return load_dataset("csv", data_files=csv_files, delimiter="\t")


def load_dataset_from_recipe(dataset_recipe_config):

    dataset_name = dataset_recipe_config.dataset_name

    main_dataset_config = load_main_dataset_config()

    # Find the dataset config using the dataset name
    dataset_config = main_dataset_config["datasets"].get(dataset_name)
    if dataset_config is None:
        print("Available datasets:")
        for available_name in main_dataset_config["datasets"]:
            print(f"  - {available_name}")
        raise ValueError(f"Dataset '{dataset_name}' not found in datasets.yaml")

    dataset_path = Path(dataset_config.get("path", ""))

    if dataset_path == "" or dataset_path is None:
        dataset_path = dataset_config["path"]
        print(
            f"Warning: Dataset path not provided or is empty. Using default path: {dataset_path}."
        )

    if "splits" in dataset_config:
        dataset = load_from_splits(dataset_path, dataset_config["splits"])
    elif "folds" in dataset_config:
        dataset = load_from_folds(dataset_path, dataset_config, dataset_recipe_config)
    else:
        raise ValueError(
            "Dataset configuration must contain either 'splits' or 'folds'."
        )

    all_labels = set()
    for split in dataset:
        all_labels.update(dataset[split][dataset_recipe_config.label_column_name])

    label_names = sorted(list(all_labels))
    if not label_names:
        raise ValueError(
            "The dataset config must include a 'labels' key with label names."
        )

    # Add ClassLabel feature for labels
    class_label = ClassLabel(names=label_names)

    # Map the labels to the ClassLabel feature
    for split in dataset:
        dataset[split] = dataset[split].cast_column(
            dataset_recipe_config.label_column_name, class_label
        )

    if dataset_recipe_config.audio_column_name not in dataset["train"].column_names:
        raise ValueError(
            f"--audio_column_name {dataset_recipe_config.audio_column_name} not found in dataset '{dataset_recipe_config.dataset_name}'. "
            "Make sure to set `--audio_column_name` to the correct audio column - one of "
            f"{', '.join(dataset['train'].column_names)}."
        )

    if dataset_recipe_config.label_column_name not in dataset["train"].column_names:
        raise ValueError(
            f"--label_column_name {dataset_recipe_config.label_column_name} not found in dataset '{dataset_recipe_config.dataset_name}'. "
            "Make sure to set `--label_column_name` to the correct text column - one of "
            f"{', '.join(dataset['train'].column_names)}."
        )

    print("=" * 20)
    print(dataset)
    total_samples = sum(len(split) for split in dataset.values())
    print(f"Total samples: {total_samples}")
    print("=" * 20)
    return dataset
