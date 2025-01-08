import os
import re
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Define consistent colors for training and validation
TRAIN_COLOR = "blue"
VAL_COLOR = "orange"

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

# Function to save results as JSON
def save_results_as_json(results, save_dir):
    for model, folds in results.items():
        model_save_path = os.path.join(save_dir, f"{model}_results.json")
        with open(model_save_path, 'w') as f:
            json.dump({fold: data for fold, data in folds.items()}, f, indent=4)
    print(f"Results have been saved as JSON files in {save_dir}")

# Function to plot and save learning curves
def save_learning_curves(results, save_dir):
    for model, folds in results.items():
        for fold, data in folds.items():
            epochs = np.arange(1, len(data["train_loss"]) + 1)
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, data["train_loss"], label="Training Loss", color=TRAIN_COLOR)
            plt.plot(epochs, data["val_loss"], label="Validation Loss", color=VAL_COLOR)
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
            plt.plot(epochs, data["train_acc"], label="Training Accuracy", color=TRAIN_COLOR)
            plt.plot(epochs, data["val_acc"], label="Validation Accuracy", color=VAL_COLOR)
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy (%)")
            plt.legend()
            plt.grid()
            fold_save_path = os.path.join(save_dir, f"{model}_accuracy_progression_fold_{fold}.png")
            plt.savefig(fold_save_path)
            plt.close()

# Function to combine folds into a single plot and place them side by side
def save_combined_plots(results, save_dir):
    for model, folds in results.items():
        # Initialize combined data
        combined_train_loss = []
        combined_val_loss = []
        combined_train_acc = []
        combined_val_acc = []
        fold_separators = []

        # Combine data from all folds
        for fold, data in sorted(folds.items()):
            start_idx = len(combined_train_loss)
            combined_train_loss.extend(data["train_loss"])
            combined_val_loss.extend(data["val_loss"])
            combined_train_acc.extend(data["train_acc"])
            combined_val_acc.extend(data["val_acc"])
            fold_separators.append(len(combined_train_loss))  # Mark where this fold ends

        # Plot combined learning curves and accuracy progression side by side
        epochs = np.arange(1, len(combined_train_loss) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Plot Learning Curves on ax1
        ax1.plot(epochs, combined_train_loss, label="Training Loss", color=TRAIN_COLOR)
        ax1.plot(epochs, combined_val_loss, label="Validation Loss", color=VAL_COLOR)
        for sep in fold_separators:
            ax1.axvline(sep, color="red", linestyle="--", linewidth=1)
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid()

        # Plot Accuracy Progression on ax2
        ax2.plot(epochs, combined_train_acc, label="Training Accuracy", color=TRAIN_COLOR)
        ax2.plot(epochs, combined_val_acc, label="Validation Accuracy", color=VAL_COLOR)
        for sep in fold_separators:
            ax2.axvline(sep, color="red", linestyle="--", linewidth=1)
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Accuracy (%)")
        ax2.legend()
        ax2.grid()

        # Save the combined figure
        combined_save_path = os.path.join(save_dir, f"{model}_combined_learning_and_accuracy.png")
        plt.tight_layout()
        plt.savefig(combined_save_path)
        plt.close()

# Main script
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process and visualize training logs.")
    parser.add_argument("run_number", type=str, help="Run number in XX format (e.g., 01, 02, 03, ...)")
    args = parser.parse_args()

    NUM_RUN = args.run_number
    file_path = f"./runs/run_{NUM_RUN}/output.txt"  # Replace with your file path
    save_dir_learning_curves = f"./runs/run_{NUM_RUN}/plots/learning_curves"
    save_dir_accuracy_progression = f"./runs/run_{NUM_RUN}/plots/accuracy_progression"
    save_dir_combined_plots = f"./runs/run_{NUM_RUN}/plots/combined_plots"
    save_dir_data = f"./runs/run_{NUM_RUN}/results"  # New directory to save parsed data

    # Create necessary directories
    os.makedirs(save_dir_learning_curves, exist_ok=True)
    os.makedirs(save_dir_accuracy_progression, exist_ok=True)
    os.makedirs(save_dir_combined_plots, exist_ok=True)
    os.makedirs(save_dir_data, exist_ok=True)

    # Parse log file
    results = parse_log(file_path)

    # Save plots
    save_learning_curves(results, save_dir_learning_curves)
    save_accuracy_progression(results, save_dir_accuracy_progression)
    save_combined_plots(results, save_dir_combined_plots)

    # Save parsed results as a JSON file
    save_results_as_json(results, save_dir_data)

    print(f"All plots and data have been saved.")

if __name__ == "__main__":
    main()
