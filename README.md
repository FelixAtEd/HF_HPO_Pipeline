# BENCHMARKING TRANSFER LEARNING IN PASSIVE SONAR: AN EVALUATION STUDY

This repository contains the boilerplate code for our paper.

While the pipeline is configured for audio datasets with HF models, it can be extended to use any dataset and models.

## Repository Structure

```
├── configs/                 # Configuration files
│   ├── example_config.yaml  # Example dataset configuration
│   ├── dataset_registery.yaml # Registry of available datasets
│   └── environment.yml      # Environment configuration
├── dataset_scripts/         # Ship-wise Enforced Spliting Strategy for ShipsEar and DeepShip
├── receipes/                # Folder containg training scripts for each methods
└── test.py                  # Model Testing script
```

## Usage

### Training with Any Method(Pretrained/Lora/TFS/LR)

To train a model using LoRA adaptation technique:

```bash
accelerate launch --gpu_ids="1" --num_processes=1 --mixed_precision=fp16 recipes/pretrained.py --config configs/dataset_config.yaml
```

## Requirements

For a complete list of dependencies, see [`configs/environment.yml`](configs/environment.yml).
