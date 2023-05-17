import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(6)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):       # ⌊(nh − kh + ph + sh)/sh⌋ × ⌊(nw − kw + pw + sw)/sw⌋.
        x = F.relu(self.bn1(self.conv1(x)))  # input(1, 28, 28) output(6, 24, 24)
        x = self.pool1(x)          # output(6, 12, 12)
        x = F.relu(self.bn2(self.conv2(x)))  # output(16, 8, 8)
        x = self.pool2(x)          # output(16, 4, 4)
        x = x.view(-1, 16*4*4)     # output(256)
        x = F.relu(self.fc1(x))    # output(120)
        x = F.relu(self.fc2(x))    # output(84)
        x = self.fc3(x)            # output(10)
        return x
