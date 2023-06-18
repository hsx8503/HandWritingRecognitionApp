import torch
import torch.nn as nn
from model import LeNet
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.datasets import  MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# 加载MNIST数据集
transform = transforms.Compose([
    transforms.ToTensor(),
])
train_set = MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_set = MNIST(root='./data', train=False, download=True, transform=transform)
val_loader = DataLoader(val_set, batch_size=16, shuffle=True)



# 定义模型，损失函数，激活函数
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = LeNet().to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

lr_scheduler = lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)

def train(train_loader, net, loss_function, optimizer):
    net.train().to(device)
    accuracy_sum, loss_sum, n = 0.0, 0.0, 0

    for step, (images, labels) in enumerate(train_loader, start=0):
        images = images.to(device)
        labels = labels.to(device)

        outputs = net(images)
        loss = loss_function(outputs, labels)
        _,predict = torch.max(outputs, dim=1)

        accuracy = torch.eq(predict, labels).sum().item() / labels.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy_sum += accuracy
        loss_sum += loss.item()
        n = n + 1

    print("Accuracy:    " + str(accuracy_sum / n))
    print("Loss:    " + str(loss_sum / n))


def val(val_loader, net, loss_function):
    net.eval()
    accuracy_sum, loss_sum, n = 0.0, 0.0, 0

    with torch.no_grad():
        for step, (images, labels) in enumerate(val_loader, start=0):
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            loss = loss_function(outputs, labels)
            _, predict = torch.max(outputs, dim=1)
            accuracy = torch.eq(predict, labels).sum().item() / labels.size(0)
            accuracy_sum += accuracy
            loss_sum += loss.item()
            n = n + 1
        print("val_accuracy:    "+str(accuracy_sum/n))
        print("val_loss:    " + str(loss_sum / n))
        return accuracy_sum / n


epoch = 30
best = 0
for epoch in range(epoch):
    print(f'epoch{epoch+1}:\n-------------------------')
    train(train_loader, net, loss_function, optimizer)
    a = val(val_loader, net, loss_function)

    if a > best:
        best = a
        save_path = 'Lenet.pth'
        torch.save(net.state_dict(), save_path)

