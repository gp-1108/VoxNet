import os
import re
import matplotlib.pyplot as plt
import numpy as np

# Function to parse the log file
def parse_log(file_path):
    results = {}
    current_fold = None
    models_dict = {}
    num_model = 0
    with open(file_path, 'r') as f:
        for line in f:

            # Detect the start of a new training run
            if "Running experiment with config" in line:
                current_fold = None
                models_dict[num_model] = None

            # Detect the start of a new fold
            fold_match = re.search(r'Fold (\d+)/\d+', line)
            if fold_match:
                current_fold = int(fold_match.group(1))
                if num_model not in results:
                    results[num_model] = {}
                results[num_model][current_fold] = {
                    "train_loss": [],
                    "train_acc": [],
                    "val_loss": [],
                    "val_acc": []
                }

            # Parse epoch data
            epoch_match = re.search(
                r'Epoch (\d+)/\d+.*Train Loss: ([\d.]+), Train Acc: ([\d.]+)% - Val Loss: ([\d.]+), Val Acc: ([\d.]+)%', line)
            if epoch_match and num_model and current_fold:
                results[num_model][current_fold]["train_loss"].append(float(epoch_match.group(2)))
                results[num_model][current_fold]["train_acc"].append(float(epoch_match.group(3)))
                results[num_model][current_fold]["val_loss"].append(float(epoch_match.group(4)))
                results[num_model][current_fold]["val_acc"].append(float(epoch_match.group(5)))

            # Detect the name of the model
            model_match = re.search(r"Evaluating .*/(.*?)\.pth on test set", line)
            if model_match:
                models_dict[num_model] = model_match.group(1) + "_folds_" + str(current_fold)
                num_model += 1

    # Create a new results object, the same as results, but change num_model to models_dict[num_model]
    new_results = {}
    for model, folds in results.items():
        if models_dict[model] is None:
            continue
        new_results[models_dict[model]] = folds
    results = new_results
    
    return results

# Function to plot and save learning curves
def save_learning_curves(results, save_dir):
    for model, folds in results.items():
        for fold, data in folds.items():
            epochs = np.arange(1, len(data["train_loss"]) + 1)
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, data["train_loss"], label="Training Loss")
            plt.plot(epochs, data["val_loss"], label="Validation Loss")
            plt.title(f"Learning Curves (Model: {model}, Fold {fold})")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid()
            fold_save_path = os.path.join(save_dir, f"{model}_learning_curve_fold_{fold}.png")
            plt.savefig(fold_save_path)
            plt.close()

# Function to plot and save accuracy progression
def save_accuracy_progression(results, save_dir):
    for model, folds in results.items():
        for fold, data in folds.items():
            epochs = np.arange(1, len(data["train_acc"]) + 1)
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, data["train_acc"], label="Training Accuracy")
            plt.plot(epochs, data["val_acc"], label="Validation Accuracy")
            plt.title(f"Accuracy Progression (Model: {model}, Fold {fold})")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy (%)")
            plt.legend()
            plt.grid()
            fold_save_path = os.path.join(save_dir, f"{model}_accuracy_progression_fold_{fold}.png")
            plt.savefig(fold_save_path)
            plt.close()

# Function to plot and save box plot for cross-validation results
def save_cross_validation_boxplot(results, save_dir):
    for model, folds in results.items():
        # Collect validation accuracies for folds that have data
        val_accs = [
            data["val_acc"][-1] for data in folds.values() if data["val_acc"]
        ]
        if not val_accs:
            print(f"Skipping box plot for model '{model}' as no validation accuracy data is available.")
            continue  # Skip if no validation accuracies are available

        plt.figure(figsize=(10, 6))
        plt.boxplot(val_accs, labels=[f"Fold {fold}" for fold, data in folds.items() if data["val_acc"]])
        plt.title(f"Cross-Validation Validation Accuracies (Model: {model})")
        plt.ylabel("Accuracy (%)")
        plt.grid()
        boxplot_save_path = os.path.join(save_dir, f"{model}_cross_validation_boxplot.png")
        plt.savefig(boxplot_save_path)
        plt.close()

# Main script
def main():
    file_path = "./run_02/output_1963420.txt"  # Replace with your file path
    save_dir = "./run_02/plots"
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

    results = parse_log(file_path)

    # Save plots
    save_learning_curves(results, save_dir)
    save_accuracy_progression(results, save_dir)

    print(f"Plots have been saved to {save_dir}")

if __name__ == "__main__":
    main()
