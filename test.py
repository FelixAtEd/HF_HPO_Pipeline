import sys
import json
import os
from pathlib import Path

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "common"))
from common.arguments import *
from common.training_pipeline import TrainingPipeline
from common.utilities import setup_logging
from common.dataset_loader import load_dataset_from_recipe

BEST_MODEL_PATH = ""  # Replace with your actual output directory

def infer_model_from_path(training_args, model_args, best_model_path):
    """Automatically configure model and task based on best_model_path."""
    MODEL_REGISTRY = {
        "facebook/wav2vec2-base": "audio_classification",
        "facebook/data2vec-audio-base": "audio_classification",
        "MIT/ast-finetuned-audioset-10-10-0.448-v2": "audio_classification",
        "Simon-Kotchou/ssast-small-patch-audioset-16-16": "audio_classification",
        "microsoft/resnet-18": "image_classification",
        "microsoft/resnet-50": "image_classification",
        "facebook/deit-small-patch16-224": "image_classification",
        "ACDNet": "image_classification",
    }

    # Try to match model path from registry using model name fragments
    for model_name, task in MODEL_REGISTRY.items():
        if model_name.replace("/", "_") in best_model_path:
            model_args.model_name_or_path = model_name
            model_args.task = task
            training_args.gradient_checkpointing = False if task == "image_classification" else training_args.gradient_checkpointing
            break
    else:
        raise ValueError("Could not infer model from best_model_path.")

    # Optional: default eval settings if needed
    training_args.eval_strategy = "no"
    training_args.load_best_model_at_end = False

    return training_args, model_args


def main():
    training_args, model_args, data_args, hpo_args = parse_arguments()

    setup_logging(training_args)

    # User selects model/config via quick launch
    training_args, model_args = infer_model_from_path(training_args, model_args,BEST_MODEL_PATH)
    raw_dataset = load_dataset_from_recipe(data_args)

    # ---------- Dataset/Configuration Setup ---------- #
    training_pipeline = TrainingPipeline(
        raw_dataset, training_args, model_args, data_args, hpo_args,model_args.model_debug
    )
    training_pipeline.debug = True
    try:
        test_output_dir = Path(BEST_MODEL_PATH).relative_to(Path(__file__).parent)
    except:
        test_output_dir = BEST_MODEL_PATH
    training_pipeline.training_args.output_dir = f"./output_single_test/{test_output_dir}"
    Path(training_pipeline.training_args.output_dir).mkdir(parents=True, exist_ok=True)


    base_model = training_pipeline.get_model(pretrained=True,trained_checkpoint=BEST_MODEL_PATH)
    print(f"Base model loaded from {BEST_MODEL_PATH}")
    if "lora" in BEST_MODEL_PATH:
        from peft import PeftModel
        base_model = PeftModel.from_pretrained(base_model, BEST_MODEL_PATH)

    test_trainer, results = training_pipeline.test_model(base_model,best_model_path=BEST_MODEL_PATH)

    id2label = test_trainer.model.config.id2label
    logits = results.predictions  # Shape (num_samples, num_classes)
    references = results.label_ids  # Shape (num_samples,)

    # Save predictions and references to JSON
    prediction_results = {
        "logits": logits.tolist(),
        "references": references.tolist(),
        "id2label": id2label,
    }

    with open(f"{training_pipeline.training_args.output_dir}/huggingface_prediction_results.json", "w") as f:
        json.dump(prediction_results, f)

if __name__ == "__main__":
    main()
