from argparse import ArgumentParser
from Dataset import ModelNet40Dataset
import models
import torch
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--model_pth", type=str, required=True)
    parser.add_argument("--model_class", type=str, required=True)
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()

    # Loading the dataset
    dataset = ModelNet40Dataset(args.dataset_dir, split="test")
    n_classes = len(dataset.get_class_mapping())

    # Loading the model
    model_class = getattr(models, args.model_class)
    model = model_class(n_classes)
    model.eval()
    model.load_state_dict(torch.load(args.model_pth, map_location='cpu'))

    # Setup environment
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rotational_axes = [(1,2),(2,3),(1,3)]
    confusion_matrix = torch.zeros(n_classes, n_classes)
    

    for i in tqdm(range(len(dataset))):
        voxel_grid, target = dataset[i]
        rotated_grids = ModelNet40Dataset.create_rotated_voxel_grids(voxel_grid, rotation_axes=rotational_axes)
        
        # Randomly select K rotations
        rotated_grids = random.sample(rotated_grids, min(args.k, len(rotated_grids)))

        final_samples = rotated_grids + [voxel_grid]

        cumulative_log_probs = torch.zeros((1,n_classes)).to(device)
        for sample in final_samples:
            sample = torch.tensor(sample).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(sample)
            probs = torch.softmax(logits, dim=1)
            cumulative_log_probs += torch.log(probs)

        # Picking the most likely class
        pred_class = torch.argmax(cumulative_log_probs).item()

        confusion_matrix[target][pred_class] += 1
    
    # Printing the accuracy
    correct = 0
    for i in range(n_classes):
        correct += confusion_matrix[i][i]
    total = confusion_matrix.sum().item()

    print(f"Accuracy: {100 * correct / total:.2f}%")

    # Saving the confusion matrix plot to disk
    plt.imshow(confusion_matrix, cmap='hot', interpolation='nearest')
    plt.savefig("confusion_matrix.png")

if __name__ == "__main__":
    main()