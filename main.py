'''
main.py

Author: Wonseok Oh
Date: May, 2023
'''

import os
import mlflow
import hydra
from omegaconf import DictConfig, OmegaConf


# This automatically reads in the configuration
@hydra.main(config_name='config')
def total_pipeline(config: DictConfig):
    '''
    total_pipeline
    Input: config (DictConfig) - config from config.yaml
    Output: None
    '''
    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # You can get the path at the root of the MLflow project with this:
    root_path = hydra.utils.get_original_cwd()
    config = OmegaConf.to_object(config)
    # Check which steps we need to execute
    if isinstance(config["main"]["execute_steps"], str):
        # This was passed on the command line as a comma-separated list of
        # steps
        steps_to_execute = config["main"]["execute_steps"].split(",")
    else:
        assert isinstance(config["main"]["execute_steps"], list)
        steps_to_execute = config["main"]["execute_steps"]

    # Download step
    if "download" in steps_to_execute:

        _ = mlflow.run(
            os.path.join(root_path, "download"),
            "main",
            parameters={
                "file_url": config["data"]["file_url"],
                "artifact_name": "raw_data.parquet",
                "artifact_type": "raw_data",
                "artifact_description": "Data as downloaded"
            },
        )

    if "preprocess" in steps_to_execute:

        _ = mlflow.run(
            uri=os.path.join(root_path, "preprocess"),
            entry_point="main",
            parameters={
                "input_artifact": "raw_data.parquet:latest",
                "artifact_name": "preprocessed_data.csv",
                "artifact_type": "preprocessed_data",
                "artifact_description": "Data with pre-processsing applied"
            }
        )

    if "check_data" in steps_to_execute:
        _ = mlflow.run(
            uri=os.path.join(root_path, "check_data"),
            entry_point="main",
            parameters={
                "reference_artifact": config["data"]["reference_dataset"],
                "sample_artifact": "preprocessed_data.csv:latest",
                "ks_alpha": config["data"]["ks_alpha"]
            }
        )

    if "segregate" in steps_to_execute:
        _ = mlflow.run(
            uri=os.path.join(root_path, "segregate"),
            entry_point="main",
            parameters={
                "input_artifact": "preprocessed_data.csv:latest",
                "artifact_root": "data",
                "artifact_type": "segregated_data",
                "test_size": config["data"]["test_size"],
                "stratify": config["data"]["stratify"]
            }
        )

    if "random_forest" in steps_to_execute:

        # Serialize decision tree configuration
        model_config = os.path.abspath("random_forest_config.yml")

        with open(model_config, "w+", encoding="utf-8") as f_p:
            f_p.write(OmegaConf.to_yaml(config["random_forest_pipeline"]))

        _ = mlflow.run(
            uri=os.path.join(root_path, "random_forest"),
            entry_point="main",
            parameters={
                "train_data": "data_train.csv:latest",
                "model_config": model_config,
                "export_artifact": config["random_forest_pipeline"]["export_artifact"],
                "random_seed": config["main"]["random_seed"],
                "val_size": config["data"]["val_size"],
                "stratify": config["data"]["stratify"]
            }
        )

    if "evaluate" in steps_to_execute:

        # YOUR CODE HERE: call the evaluate step
        _ = mlflow.run(
            uri=os.path.join(root_path, "evaluate"),
            entry_point="main",
            parameters={
                "model_export": f"{config['random_forest_pipeline']['export_artifact']}:latest",
                "test_data": "data_test.csv:latest"
            }
        )


if __name__ == "__main__":
    total_pipeline()
