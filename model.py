import torch.nn as nn

DROPOUT_PROB = 0.1

class CIFAR10Net(nn.Module):
    def __init__(self):
        super(CIFAR10Net, self).__init__()

        # BLOCK 1
        # Input: 32x32x3 | Output: 32x32x32 | RF: 5
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(DROPOUT_PROB),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(DROPOUT_PROB)
        )

        # BLOCK 2
        # Input: 32x32x32 | Output: 16x16x32 | RF: 7
        self.conv_block2 = nn.Sequential(
             nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
             nn.ReLU(),
             nn.BatchNorm2d(32)
        )

        # BLOCK 3
        # Input: 16x16x32 | Output: 16x16x64 | RF: 31
        self.conv_block3 = nn.Sequential(
            # Depthwise Separable Convolution
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, groups=32, bias=False), # Depthwise
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, bias=False), # Pointwise
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(DROPOUT_PROB),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(DROPOUT_PROB),
        )

        # BLOCK 4
        # Input: 16x16x64 | Output: 8x8x128 | RF: 39
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )

        # OUTPUT BLOCK
        # Input: 8x8x128 | Output: 1x1x10 | RF: 55
        self.output_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # GAP layer
            nn.Conv2d(in_channels=128, out_channels=10, kernel_size=1, bias=False)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.output_block(x)
        x = x.view(-1, 10)
        return x