from torch import nn

class BaseVoxNet(nn.Module):
    def __init__(self, n_classes, voxel_dim):
        super().__init__()
        self.n_classes = n_classes
        # Convolutional layers
        self.conv1 = nn.Conv3d(1, voxel_dim, kernel_size=5, stride=2)
        self.conv2 = nn.Conv3d(32, 32, kernel_size=3, stride=1)
        # Pooling layer
        self.pool = nn.MaxPool3d(2)
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 6 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, n_classes)
        # Dropout layer
        self.dropout = nn.Dropout(0.5)
        # Activation
        self.relu = nn.ReLU()

    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        print(x.shape)
        x = self.relu(x)
        # Second conv block
        x = self.conv2(x)
        print(x.shape)
        x = self.relu(x)
        # Pooling
        x = self.pool(x)
        print(x.shape)
        # Flatten
        x = x.flatten(1)
        # First fc layer
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        # Output layer
        x = self.fc2(x)

        return x
