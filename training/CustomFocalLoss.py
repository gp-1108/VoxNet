import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomFocalLoss(nn.Module):
    def __init__(self, gamma=2, occurences: dict = None, mode="mean", device=torch.device("cpu")):
        super(CustomFocalLoss, self).__init__()
        self.gamma = gamma
        self.occurences = occurences
        self.mode = mode
        self.device = device

        # Computing the alpha values
        total_size = sum(occurences.values())
        self.alpha = {k: total_size / v for k, v in occurences.items()}
        self.alpha_tensor = torch.tensor(
            [self.alpha[k] for k in sorted(self.alpha.keys())],
            dtype=torch.float32
        )  # Create a tensor for alpha values in class order
        self.alpha_tensor = self.alpha_tensor.to(self.device)
        if self.mode not in ["mean", "sum", "none"]:
            raise ValueError("mode must be one of 'mean', 'sum', or 'none'")

    def forward(self, batch_logits, targets):
        # Get softmax probabilities for all classes
        probs = F.softmax(batch_logits, dim=-1)

        # Gather the probabilities corresponding to the target class
        target_probs = probs[torch.arange(len(targets)), targets]

        # Map the target classes to their respective alpha values
        alphas = self.alpha_tensor[targets]

        # Compute the focal loss
        loss = -alphas * (1 - target_probs) ** self.gamma * torch.log(target_probs)

        if self.mode == "sum":
            # Return the sum of the loss over the batch
            return loss.sum()
        elif self.mode == "none":
            # Return the loss for each sample in the batch
            return loss
        else:
            # Return the mean loss over the batch
            return loss.mean()
