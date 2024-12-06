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

# # Function to plot and save box plot for cross-validation results
# def save_cross_validation_boxplot(results, save_dir):
#     for model, folds in results.items():
#         # Collect validation accuracies for folds that have data
#         val_accs = [
#             data["val_acc"][-1] for data in folds.values() if data["val_acc"]
#         ]
#         if not val_accs:
#             print(f"Skipping box plot for model '{model}' as no validation accuracy data is available.")
#             continue  # Skip if no validation accuracies are available

#         plt.figure(figsize=(10, 6))
#         plt.boxplot(val_accs, labels=[f"Fold {fold}" for fold, data in folds.items() if data["val_acc"]])
#         plt.title(f"Cross-Validation Validation Accuracies (Model: {model})")
#         plt.ylabel("Accuracy (%)")
#         plt.grid()
#         boxplot_save_path = os.path.join(save_dir, f"{model}_cross_validation_boxplot.png")
#         plt.savefig(boxplot_save_path)
#         plt.close()

# Function to combine folds into a single plot
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

        # Plot combined learning curves
        epochs = np.arange(1, len(combined_train_loss) + 1)
        plt.figure(figsize=(12, 8))
        plt.plot(epochs, combined_train_loss, label="Training Loss", color="blue")
        plt.plot(epochs, combined_val_loss, label="Validation Loss", color="orange")
        for sep in fold_separators:
            plt.axvline(sep, color="red", linestyle="--", linewidth=1)
        plt.title(f"Combined Learning Curves (Model: {model})")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        combined_loss_save_path = os.path.join(save_dir, f"{model}_combined_learning_curves.png")
        plt.savefig(combined_loss_save_path)
        plt.close()

        # Plot combined accuracy progression
        plt.figure(figsize=(12, 8))
        plt.plot(epochs, combined_train_acc, label="Training Accuracy", color="green")
        plt.plot(epochs, combined_val_acc, label="Validation Accuracy", color="purple")
        for sep in fold_separators:
            plt.axvline(sep, color="red", linestyle="--", linewidth=1)
        plt.title(f"Combined Accuracy Progression (Model: {model})")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.grid()
        combined_acc_save_path = os.path.join(save_dir, f"{model}_combined_accuracy_progression.png")
        plt.savefig(combined_acc_save_path)
        plt.close()


# Main script
def main():
    NUM_RUN = "01" # XX format (g.e. 01, 02, 03, ...)
    file_path = f"./runs/run_{NUM_RUN}/output.txt"  # Replace with your file path
    save_dir_learning_curves = f"./runs/run_{NUM_RUN}/plots/learning_curves"
    save_dir_accuracy_progression = f"./runs/run_{NUM_RUN}/plots/accuracy_progression"
    save_dir_combined_plots = f"./runs/run_{NUM_RUN}/plots/combined_plots"
    os.makedirs(save_dir_learning_curves, exist_ok=True)  # Create the directory if it doesn't exist
    os.makedirs(save_dir_accuracy_progression, exist_ok=True)
    os.makedirs(save_dir_combined_plots, exist_ok=True)

    results = parse_log(file_path)

    # Save plots
    save_learning_curves(results, save_dir_learning_curves)
    save_accuracy_progression(results, save_dir_accuracy_progression)
    save_combined_plots(results, save_dir_combined_plots)

    print(f"All plots have been saved")

if __name__ == "__main__":
    main()
