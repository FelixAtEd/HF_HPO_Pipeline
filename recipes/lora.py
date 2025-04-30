import sys
import yaml
import os
from pathlib import Path

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "common"))
from peft import LoraConfig, get_peft_model, PeftModel

from common.arguments import *
from common.training_pipeline import TrainingPipeline
from common.utilities import setup_logging, quick_launch, print_trainable_parameters
from common.dataset_loader import load_dataset_from_recipe


def main():
    training_args, model_args, data_args, hpo_args = parse_arguments()

    setup_logging(training_args)

    # User selects model/config via quick launch
    # training_args, model_args = quick_launch(training_args, model_args)

    raw_dataset = load_dataset_from_recipe(data_args)

    training_pipeline = TrainingPipeline(
        raw_dataset, training_args, model_args, data_args, hpo_args,model_args.model_debug
    )

    training_pipeline.create_output_dir("lora")
    if training_args.eval_strategy != "no":
        def model_init():
            model = training_pipeline.get_model(pretrained=True)
            target_modules = ["q_proj", "v_proj","k_proj", "query", "value","key","convolution"]
            modules_to_save = ["classifier", "projector","normalization"]
            peft_config = LoraConfig(
                inference_mode=False,
                r=8,
                target_modules=target_modules,
                modules_to_save=modules_to_save,
                lora_alpha=32,
                lora_dropout=0.1,
            )
            model = get_peft_model(model, peft_config)
            return model
        
        if model_args.model_debug:
            model = model_init()
            for name, param in model.named_parameters():
                print(f"{name}:{param.requires_grad}")     
            print_trainable_parameters(model)
            input("Press Enter to continue...")
            
        training_pipeline.create_trainer(model_init=model_init)

        training_pipeline.run_trainer_HPO()


    # For wav2vec2 and data2vec, when trying to load a checkpoint,
    # the default add_adpater class is implmented with ASR in mind and will not run.
    # You can add pass to the add_adpater class to skip it as adpaters will get loaded via PeftModel
    best_model_path = os.path.join(training_pipeline.training_args.output_dir, "_best")

    base_model = training_pipeline.get_model(pretrained=True,trained_checkpoint=best_model_path)
    lora_model = PeftModel.from_pretrained(base_model, best_model_path)

    training_pipeline.test_model(lora_model,best_model_path)


if __name__ == "__main__":
    main()
