from torch import nn

class BaseVoxNet(nn.Module):
    def __init__(self, n_classes, voxel_dim):
        super().__init__()
        self.n_classes = n_classes
        self.net = nn.Sequential(
            nn.Conv3d(1, voxel_dim, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Flatten(),
            nn.Linear(32 * 6 * 6 * 6, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        return self.net(x)
