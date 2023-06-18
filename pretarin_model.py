import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import USPS
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


#实现残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

#定义预训练模型
class PretrainModel(nn.Module):
    def __init__(self, num_classes):
        super(PretrainModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            ResidualBlock(32, 64, stride=1),
            nn.MaxPool2d(kernel_size=2),
            ResidualBlock(64, 64, stride=1),  # 第二个残差块
            nn.MaxPool2d(kernel_size=2),
        ResidualBlock(64, 64, stride=1)  # 第三个残差块
        )
        self.fc = nn.Linear(576, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        y = self.fc(x)
        return y


# 加载USPS数据集
usps_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

usps_dataset = USPS(root='./data', train=True, transform=usps_transform, download=True)
usps_loader = DataLoader(dataset=usps_dataset, batch_size=16, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_function = nn.CrossEntropyLoss()
pretrained_path = './Lenet.pth'

model = PretrainModel(10).to(device)

# 定义损失函数和优化器
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

def load_pretrained_lenet(model, pretrained_path):
    pretrained_dict = torch.load(pretrained_path)
    model_dict = model.state_dict()
    # 剔除不匹配的键值对
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 更新模型的权重参数
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

def train_transfer_model(model, train_loader, loss_function, optimizer):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for step, (images, labels) in enumerate(train_loader, start=0):
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = loss_function(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, dim=1)
        accuracy = torch.eq(predicted, labels).sum().item() / labels.size(0)

        total = total + 1
        correct += accuracy

        train_loss = running_loss / total
        train_acc = correct / total

        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        return train_acc

if __name__ == '__main__':
    load_pretrained_lenet(model, pretrained_path)
    # 训练迁移学习模型
    num_epochs = 70
    best = 0
    for epoch in range(num_epochs):
        print(f'epoch{epoch + 1}:\n-------------------------')
        a = train_transfer_model(model, usps_loader, loss_function, optimizer)

        if a > best:
            best = a
            save_path = 'Lenet-new.pth'
            torch.save(model.state_dict(), save_path)