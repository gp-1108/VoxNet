from torch import nn

class ResVoxBlock(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv1 = nn.Conv3d(input_dim, input_dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(input_dim, input_dim, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()


    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x += identity
        x = self.relu(x)
        return x

class ResVoxNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

        # Convolutional layers
        self.conv1 = nn.Conv3d(1, 30, kernel_size=5, stride=2, padding=3)
        self.conv2 = nn.Conv3d(30, 7, kernel_size=4, stride=1, padding=0)

        # Residual blocks
        self.res_block1 = ResVoxBlock(30)
        self.res_block2 = ResVoxBlock(30)
        self.res_block3 = ResVoxBlock(30)

        # Pooling layer
        self.pool = nn.MaxPool3d(3)

        # Fully connected layers
        self.fc1 = nn.Linear(7 * 4 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, n_classes)

        # Misc
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)



    def forward(self, x):
        # First conv block 1x32x32x32 -> 30x17x17x17
        x = self.conv1(x)
        x = self.relu(x)

        # Residual blocks 30x17x17x17 -> 30x17x17x17
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)

        # Second conv block  30x17x17x17 -> 7x14x14x14
        x = self.conv2(x)
        x = self.relu(x)

        # Time for pooling 7x14x14x14 -> 7x4x4x4
        x = self.pool(x)

        # Flatten 7x4x4x4 -> 448
        x = x.flatten(1)

        # Now time for the fully connected layers
        # First FC layer 448 -> 128
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        # Output layer 128 -> n_classes
        x = self.fc2(x)

        return x

