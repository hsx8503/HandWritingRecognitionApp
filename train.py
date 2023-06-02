import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms

# 加载数据集，使用MNIST
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.5], std=[0.5])]
)
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)
val_set = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=False, num_workers=0)

# 定义模型，损失函数，激活函数
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = LeNet()
net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)


def train(train_loader, net, loss_function, optimizer):
    net.train()
    accuracy_sum, loss_sum, n = 0.0, 0.0, 0

    for step, (images, labels) in enumerate(train_loader, start=0):
        outputs = net(images)
        loss = loss_function(outputs, labels)
        predict = torch.max(outputs, dim=1)[1]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n = n + 1
        accuracy = torch.eq(predict, labels).sum().item() / labels.size(0)
        accuracy_sum += accuracy
        loss_sum += loss.item()

    print("Accuracy:    " + str(accuracy_sum / n))
    print("Loss:    " + str(loss_sum / n))


def val(val_loader, net, loss_function, optimizer):
    net.eval()
    accuracy_sum, loss_sum, n = 0.0, 0.0, 0

    with torch.no_grad():
        for step, (images, labels) in enumerate(val_loader, start=0):
            outputs = net(images)
            loss = loss_function(outputs, labels)
            predict = torch.max(outputs, dim=1)[1]
            n = n + 1
            accuracy = torch.eq(predict, labels).sum().item() / labels.size(0)
            accuracy_sum += accuracy
            loss_sum += loss.item()
        print("val_accuracy:    "+str(accuracy_sum/n))
        return accuracy_sum / n


epoch = 30
best = 0
for epoch in range(epoch):
    print(f'epoch{epoch+1}:\n')
    train(train_loader, net, loss_function, optimizer)
    a = val(val_loader, net, loss_function, optimizer)

    if a > best:
        best = a
        save_path = './Lenet06.pth'
        torch.save(net.state_dict(), save_path)
