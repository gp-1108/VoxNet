from torch import nn

class BaseVoxNet(nn.Module):
    def __init__(self, n_classes, voxel_dim):
        super().__init__()
        self.n_classes = n_classes
        # convolutional layers
        self.conv1 = nn.conv3d(1, voxel_dim, kernel_size=5, stride=2)
        self.conv2 = nn.conv3d(32, 32, kernel_size=3, stride=1)
        # pooling layer
        self.pool = nn.maxpool3d(2)
        # fully connected layers
        self.fc1 = nn.linear(32 * 6 * 6 * 6, 128)
        self.fc2 = nn.linear(128, n_classes)
        # dropout layer
        self.dropout = nn.dropout(0.5)
        # activation
        self.relu = nn.relu()

    def forward(self, x):
        # first conv block
        x = self.conv1(x)
        print(x.shape)
        x = self.relu(x)
        # second conv block
        x = self.conv2(x)
        print(x.shape)
        x = self.relu(x)
        # pooling
        x = self.pool(x)
        print(x.shape)
        # flatten
        x = x.flatten(1)
        # first fc layer
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        # output layer
        x = self.fc2(x)

        return x
