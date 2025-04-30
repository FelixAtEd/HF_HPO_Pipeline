import sys
import logging

import transformers

logger = logging.getLogger(__name__)


def setup_logging(training_args):
    """Set up logging configuration."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)
    transformers.set_seed(42)

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


def print_trainable_parameters(model):

    trainable_params = sum(
        p.numel()
        for n, p in model.named_parameters()
        if p.requires_grad and "classifier" not in n
    )
    all_params = sum(
        p.numel() for n, p in model.named_parameters() if "classifier" not in n
    )
    trainable_percent = 100 * trainable_params / all_params

    print(
        f"trainable params: {trainable_params:,} || all params: {all_params:,} || trainable%: {trainable_percent:.4f}"
    )


# Overide the default model path and task for easy iteration
def quick_launch(training_args, model_args):
    """Configure model and training mode through user selection."""
    MODEL_REGISTRY = {
        "1": {
            "model_name_or_path": "facebook/wav2vec2-base",
        },
        "2": {
            "model_name_or_path": "facebook/data2vec-audio-base",
        },
        "3": {
            "model_name_or_path": "MIT/ast-finetuned-audioset-10-10-0.448-v2",
        },
        "4": {
            "model_name_or_path": "Simon-Kotchou/ssast-small-patch-audioset-16-16",
        },
        "5": {
            "model_name_or_path": "microsoft/resnet-18",
        },
        "6": {
            "model_name_or_path": "microsoft/resnet-50",
        },
        "7": {
            "model_name_or_path": "facebook/data2vec-vision-base",
        },
    }

    # Mode selection
    print("Select Mode:")
    print("[1] HPO training")
    print("[2] Test only")
    mode = input("Enter mode (1/2): ").strip()

    # Update evaluation strategy based on mode
    training_args.eval_strategy = training_args.eval_strategy = (
        "steps" if mode == "1" else "no"
    )
    training_args.load_best_model_at_end = True if mode == "1" else False
    # Model selection
    print("\nAvailable Models:")
    for key, config in MODEL_REGISTRY.items():
        print(f"[{key}] {config['model_name_or_path']}\n")

    model_choice = input("Select model: ").strip()
    selected = MODEL_REGISTRY.get(model_choice)
    if int(model_choice) in [5, 6, 7]:
        model_args.task = "image_classification"
    else:
        model_args.task = "audio_classification"
    if int(model_choice) in [5, 6, 7]:
        training_args.gradient_checkpointing = False
    model_args.model_name_or_path = selected["model_name_or_path"]
    return training_args, model_args
