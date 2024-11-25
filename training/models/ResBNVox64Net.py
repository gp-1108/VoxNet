from torch import nn

class ResVoxBlock(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv1 = nn.Conv3d(input_dim, input_dim, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(input_dim)
        self.conv2 = nn.Conv3d(input_dim, input_dim, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += identity
        x = self.relu(x)
        return x


class ResBNVox64Net(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

        # Convolutional layers
        self.conv1 = nn.Conv3d(1, 30, kernel_size=7, stride=2, padding=1)
        self.bn1 = nn.BatchNorm3d(30)
        self.conv2 = nn.Conv3d(30, 30*2, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(30*2)
        self.conv3 = nn.Conv3d(30*2, 30*4, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(30*4)

        # Residual blocks
        self.res_block1 = ResVoxBlock(30)
        self.res_block2 = ResVoxBlock(30)
        self.res_block3 = ResVoxBlock(30)
        self.res_block4 = ResVoxBlock(30*2)
        self.res_block5 = ResVoxBlock(30*2)
        self.res_block6 = ResVoxBlock(30*2)
        self.res_block7 = ResVoxBlock(30*2)
        self.res_block8 = ResVoxBlock(30*2)
        self.res_block9 = ResVoxBlock(30*4)
        self.res_block10 = ResVoxBlock(30*4)
        self.res_block11 = ResVoxBlock(30*4)
        self.res_block12 = ResVoxBlock(30*4)
        self.res_block13 = ResVoxBlock(30*4)
        self.res_block14 = ResVoxBlock(30*4)
        self.res_block15 = ResVoxBlock(30*4)
        self.res_block16 = ResVoxBlock(30*4)

        # Fully connected layers
        self.fc1 = nn.Linear(120 * 3 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, n_classes)

        # Misc
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.avg_pool = nn.AvgPool3d(kernel_size=3, stride=2)

    def forward(self, x):
        # First conv block 1x64x64x64 -> 30x30x30x30
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Residual blocks 30x30x30x30 -> 30x30x30x30
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)

        # Second conv block 30x30x30x30 -> 60x15x15x15
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # Residual blocks 60x15x15x15 -> 60x15x15x15
        x = self.res_block4(x)
        x = self.res_block5(x)
        x = self.res_block6(x)
        x = self.res_block7(x)
        x = self.res_block8(x)

        # Third conv block 60x15x15x15 -> 120x8x8x8
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        # Residual blocks 120x8x8x8 -> 120x8x8x8
        x = self.res_block9(x)
        x = self.res_block10(x)
        x = self.res_block11(x)
        x = self.res_block12(x)
        x = self.res_block13(x)
        x = self.res_block14(x)
        x = self.res_block15(x)
        x = self.res_block16(x)

        # Time for pooling 120x8x8x8 -> 120x3x3x3
        x = self.avg_pool(x)

        # Flatten 120x3x3x3 -> 3240
        x = x.flatten(1)

        # Now time for the fully connected layers
        # First FC layer 3240 -> 512
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        # Output layer 512 -> n_classes
        x = self.fc2(x)

        return x
