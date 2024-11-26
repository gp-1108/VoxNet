import os
import itertools
import argparse
import torch
from Dataset import ModelNet40Dataset
from models import BaseVoxNet, BatchNormVoxNet, ResVoxNet, ResBNVoxNet
from Trainer import Trainer

def get_instance(class_name, module, **kwargs):
    """
    Dynamically instantiates a class or function from a module.
    """
    cls = getattr(module, class_name)
    return cls(**kwargs)

def generate_model_name(config, model_type):
    """
    Generate a descriptive model name based on parameters.
    """
    dataset_mode = config.get("dataset_mode", "none")
    optimizer_name = config["optimizer"]["name"]
    lr = config["optimizer"]["params"]["lr"]
    loss_name = config["loss"]["name"]
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]

    return f"{model_type}_bs{batch_size}_epochs{num_epochs}_{optimizer_name}_lr{lr:.0e}_{loss_name}_{dataset_mode}.pth"

def run_experiment(dataset_path, output_path, config):
    """
    Runs a single experiment based on the given configuration.
    """
    print(f"Running experiment with config: {config}")

    # Load datasets
    train_dataset = ModelNet40Dataset(dataset_path, split="train")
    test_dataset = ModelNet40Dataset(dataset_path, split="test")

    # Apply dataset mode if specified
    if config.get("dataset_mode"):
        train_dataset.augment_voxel_grid(config["dataset_mode"])

    # Initialize model
    n_classes = len(train_dataset.get_class_mapping())
    # Test each model type
    model_types = {
        'base': BaseVoxNet,
        'batchnorm': BatchNormVoxNet,
        'res': ResVoxNet,
        'resbn': ResBNVoxNet,
    }

    for model_type, model_class in model_types.items():
        # Generate model name
        model_name = generate_model_name(config, model_type)

        # Initialize model
        model = model_class(n_classes)

        # Initialize optimizer and loss function
        optimizer = get_instance(
            config["optimizer"]["name"],
            torch.optim,
            params=model.parameters(),
            **config["optimizer"]["params"]
        )
        loss_fn = get_instance(
            config["loss"]["name"],
            torch.nn,
            **config["loss"].get("params", {})
        )

        # Define device and Trainer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        output_model_path = os.path.join(output_path, model_name)
        trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            optimizer=optimizer,
            loss_fn=loss_fn,
            output_path=output_model_path,
            device=device,
            k_folds=config["k_folds"],
        )

        # Train the model
        trainer.train(num_epochs=config["num_epochs"], 
                     batch_size=config["batch_size"], 
                     log_interval=50)

def main(args):
    # Define parameter grid for grid search
    param_grid = {
        "batch_size": [32, 64, 128, 256],
        "num_epochs": [50, 100, 150, 200],
        "k_folds": [3, 4, 5, 6],
        "optimizer": [
            {"name": "Adam", "params": {"lr": lr}} for lr in [1e-3, 5e-4, 1e-4]
        ],
        "loss": [
            {"name": "CrossEntropyLoss", "params": {}},
            {"name": "NLLLoss", "params": {}}
        ],
        "dataset_mode": [None, "v_rotate", "full_rotate", "augmented"]
    }

    # Generate all combinations of parameters
    keys, values = zip(*param_grid.items())
    experiment_configs = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Ensure output folder exists
    os.makedirs(args.output_path, exist_ok=True)

    # Run each experiment
    for config in experiment_configs:
        run_experiment(args.dataset_path, args.output_path, config)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train a VoxNet model on ModelNet40 dataset")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the ModelNet40 dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the trained models")
    args = parser.parse_args()
    main(args)
