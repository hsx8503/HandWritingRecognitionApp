import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from model import LeNet
from torch.autograd import Variable
import pickle
import cv2
import matplotlib.pyplot as plt

# 加载数据集，使用MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
])
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)
val_set = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=False, num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = LeNet().to(device)
state_dict = torch.load('G:/HandWritingRecognizationApp/Lenet3.pth', map_location=torch.device(device))
net.load_state_dict(state_dict)
net.eval()

classes = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9"
]
#
# show = transforms.ToPILImage()
#
# for i in range(20):
#     X,y=val_set[i][0],val_set[i][1]
#     show(X).show()
#
#     X = Variable(torch.unsqueeze(X,dim=0).float()).to(device)
#     with torch.no_grad():
#         pred = net(X)
#         predicted,actual = classes[torch.argmax(pred[0])],classes[y]
#         print(f'predicted:"{predicted}",actual:"{actual}"')

if __name__ == '__main__':
    # 读取图像文件
    image = Image.open('./imgs/5.png').convert('L')

    # 定义图像变换
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image_tensor = transforms.ToTensor()(image)
    image.show()

    image_tensor = Variable(torch.unsqueeze(image_tensor,dim=0).float()).to(device)
    with torch.no_grad():
        pred = net(image_tensor)
        predicted = classes[torch.argmax(pred[0])]

    # 输出识别结果
    print(f"Predicted digit: {predicted}")