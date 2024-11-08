import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import time

class Trainer:
    def __init__(self,
                 model,
                 train_dataset,
                 test_dataset,
                 optimizer,
                 loss_fn,
                 device=None,
                 output_path="model.pth",
                 k_folds=5):
        """
        Initializes the trainer with model, dataset, test data loader, optimizer, loss function, device, and k-folds.
        """
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.k_folds = k_folds
        self.output_path = output_path

    def train(self, num_epochs, batch_size=32, log_interval=10):
        """
        Main training loop using k-fold cross-validation.
        """
        kfold = KFold(n_splits=self.k_folds, shuffle=True)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

        for fold, (train_idx, val_idx) in enumerate(kfold.split(self.train_dataset)):
            print(f'Fold {fold + 1}/{self.k_folds}')

            # Create data loaders for the current fold
            train_subset = Subset(self.train_dataset, train_idx)
            val_subset = Subset(self.train_dataset, val_idx)
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

            for epoch in range(1, num_epochs + 1):
                start_time = time.time()
                train_loss, train_accuracy = self._train_one_epoch(train_loader, epoch, log_interval)
                val_loss, val_accuracy = self._validate(val_loader, epoch)
                end_time = time.time()

                print(f"Fold {fold + 1}, Epoch {epoch}/{num_epochs} - "
                      f"Time: {end_time - start_time:.2f}s - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}% - "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

        # Evaluate on the test set after training on all folds
        print("Evaluating on test set...")
        test_loss, test_accuracy = self._validate(test_loader, final_test=True)
        print(f"Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

        # Save the model
        torch.save(self.model.state_dict(), self.output_path)
        print(f"Model saved to {self.output_path}")

    def _train_one_epoch(self, train_loader, epoch, log_interval):
        """
        Trains the model for one epoch and returns the average loss and accuracy.
        """
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == targets).sum().item()
            total_samples += targets.size(0)

            if batch_idx % log_interval == 0:
                print(f"Train Epoch: {epoch} [{batch_idx * len(inputs)}/{len(train_loader.dataset)} "
                      f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

        avg_loss = running_loss / total_samples
        accuracy = 100. * correct_predictions / total_samples
        return avg_loss, accuracy

    def _validate(self, data_loader, epoch=None, final_test=False):
        """
        Validates the model and returns the average loss and accuracy. 
        """
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == targets).sum().item()
                total_samples += targets.size(0)

        avg_loss = running_loss / total_samples
        accuracy = 100. * correct_predictions / total_samples
        if not final_test:
            print(f"Validation Epoch {epoch} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        return avg_loss, accuracy
