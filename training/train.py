import os
import argparse
import torch
from Dataset import ModelNet40Dataset
from BaseModel import BaseVoxNet
from Trainer import Trainer

def get_instance(class_name, module, **kwargs):
    """
    Dynamically instantiates a class or function from a module.
    """
    cls = getattr(module, class_name)
    return cls(**kwargs)

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
    model = BaseVoxNet(n_classes, 32)

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
    output_model_path = os.path.join(output_path, config["model_name"])
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
    trainer.train(num_epochs=config["num_epochs"], batch_size=config["batch_size"], log_interval=50)

def main(args):
    # Experiment configurations
    experiments = [
        {
            "model_name": "k3fold_epoch50_es_adam_lr1e-3_celoss_b128.pth",
            "batch_size": 128,
            "num_epochs": 50,
            "k_folds": 3,
            "optimizer": {
                "name": "Adam",
                "params": {
                    "lr": 1e-3,
                },
            },
            "loss": {
                "name": "CrossEntropyLoss",
                "params": {}
            },
        },
        {
            "model_name": "k3fold_epoch50_es_adam_lr1e-3_celoss_b128_vrotation.pth",
            "batch_size": 128,
            "num_epochs": 50,
            "k_folds": 3,
            "optimizer": {
                "name": "Adam",
                "params": {
                    "lr": 1e-3,
                },
            },
            "loss": {
                "name": "CrossEntropyLoss",
                "params": {}
            },
            "dataset_mode": "v_rotate",
        },
        {
            "model_name": "k3fold_epoch50_es_SGD-3_celoss_b128_vrotation.pth",
            "batch_size": 128,
            "num_epochs": 50,
            "k_folds": 3,
            "optimizer": {
                "name": "SGD",
                "params": {
                    "lr": 1e-3,
                },
            },
            "loss": {
                "name": "CrossEntropyLoss",
                "params": {}
            },
            "dataset_mode": "v_rotate",
        },
        {
            "model_name": "k3fold_epoch50_es_SGD-3_celoss_b128.pth",
            "batch_size": 128,
            "num_epochs": 50,
            "k_folds": 3,
            "optimizer": {
                "name": "SGD",
                "params": {
                    "lr": 1e-3,
                },
            },
            "loss": {
                "name": "CrossEntropyLoss",
                "params": {}
            },
        },
        # Add more configurations as needed
    ]

    # Ensure output folder exists
    os.makedirs(args.output_path, exist_ok=True)

    # Run each experiment
    for config in experiments:
        run_experiment(args.dataset_path, args.output_path, config)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train a VoxNet model on ModelNet40 dataset")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the ModelNet40 dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the trained models")
    args = parser.parse_args()
    main(args)
