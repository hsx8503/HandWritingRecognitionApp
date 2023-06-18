import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
from bayes_opt import BayesianOptimization
from model import LeNet
import torch.optim.lr_scheduler as lr_scheduler

# 加载数据集，使用MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
])
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)
val_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=16, shuffle=True)

# 定义模型，损失函数，激活函数
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# net = LeNet().to(device)
# loss_function = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)


def train(train_loader, net, loss_function, optimizer):
    net.train()
    accuracy_sum, loss_sum, n = 0.0, 0.0, 0

    for step, (images, labels) in enumerate(train_loader, start=0):
        images = images.to(device)
        labels = labels.to(device)

        outputs = net(images)
        loss = loss_function(outputs, labels)
        _,predict = torch.max(outputs, axis=1)

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


def target(lr, momentum, lr_scheduler=lr_scheduler):
    # 定义模型，损失函数，激活函数
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = LeNet().to(device)  #实例化模型并移动到GPU
    loss_function = nn.CrossEntropyLoss()  #使用交叉熵损失函数
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum) #使用随机梯度下降算法优化参数

    # 训练模型
    lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) #每10个epoch将学习率乘以0.1
    epoch = 30
    best = 0
    for epoch in range(epoch):
        train(train_loader, net, loss_function, optimizer)
        val_acc = val(val_loader, net, loss_function)
        if val_acc > best:
            best = val_acc
            save_path = './Lenet3.pth'
            torch.save(net.state_dict(), save_path)
        lr_scheduler.step()

    return val_acc
# 定义超参数的搜索范围
pbounds = {'lr': (1e-5, 1e-1), 'momentum': (0.0, 1.0)} #定义超参数的搜索范围

# 运行贝叶斯优化
optimizer = BayesianOptimization(
    f=target, #优化目标函数
    pbounds=pbounds, #超参数搜索范围
    random_state=42, #随机数种子
    verbose=2, #输出详细信息
)

optimizer.maximize(n_iter=1) #运行贝叶斯优化

print(optimizer.max)

# 定义超参数的搜索范围