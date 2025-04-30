import sys
import yaml
import os
from pathlib import Path

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "common"))
from common.arguments import *
from common.training_pipeline import TrainingPipeline
from common.utilities import setup_logging, quick_launch,compare_weights
from common.dataset_loader import load_dataset_from_recipe

def main():
    training_args, model_args, data_args, hpo_args = parse_arguments()

    # Setup logging early for proper tracking
    setup_logging(training_args)

    # User selects model/config via quick launch
    # training_args, model_args = quick_launch(training_args, model_args)

    raw_dataset = load_dataset_from_recipe(data_args)

    # ---------- Dataset/Configuration Setup ---------- #
    training_pipeline = TrainingPipeline(
        raw_dataset, training_args, model_args, data_args, hpo_args,model_args.model_debug
    )


    training_pipeline.create_output_dir("tfs")

    if training_args.eval_strategy != "no":
        # ---------- Model Setup ---------- #
        def model_init():
            model = training_pipeline.get_model(pretrained=False)
            return model

        # Compare the weights of the two models
        if model_args.model_debug:
            model = model_init()
            for name, param in model.named_parameters():
                print(f"{name}:{param.requires_grad}")     

        training_pipeline.create_trainer(model_init=model_init)

        training_pipeline.run_trainer_HPO()


    best_model_path = os.path.join(training_pipeline.training_args.output_dir, "_best")

    base_model = training_pipeline.get_model(pretrained=True,trained_checkpoint=best_model_path)

    training_pipeline.test_model(base_model,best_model_path)



if __name__ == "__main__":
    main()
