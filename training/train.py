import argparse
import torch
from Dataset import ModelNet40Dataset
from BaseModel import BaseVoxNet
from Trainer import Trainer

def main(args):
    # Load datasets
    train_dataset = ModelNet40Dataset(args.dataset_path, split="train")
    test_dataset = ModelNet40Dataset(args.dataset_path, split="test")

    # Initialize model, optimizer, and loss function
    n_classes = len(train_dataset.get_class_mapping())
    model = BaseVoxNet(n_classes, 32)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Define device and Trainer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        optimizer=optimizer,
        loss_fn=loss_fn,
        output_path=args.output_path,
        device=device,
        k_folds=2,
    )

    # Train the model
    trainer.train(num_epochs=args.num_epochs, batch_size=args.batch_size, log_interval=50)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train a VoxNet model on ModelNet40 dataset")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the ModelNet40 dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the trained model")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs to train")

    args = parser.parse_args()
    main(args)
