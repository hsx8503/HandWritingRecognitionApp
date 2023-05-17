import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from model import LeNet
import pickle
import cv2
import matplotlib.pyplot as plt


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LeNet()
    # 从文件中读取模型
    state_dict = torch.load('Lenet04.pth', map_location=torch.device(device))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    # 读取图像文件
    image = Image.open('./imgs/3.png').convert('L')
    image_invert = ImageOps.invert(image)
    image.show()
    # 定义图像变换
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # 进行图像变换并增加batch维度
    image_tensor = transform(image).unsqueeze(0).to(device)
    # 使用模型进行推理
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)

    # 输出识别结果
    print(f"Predicted digit: {predicted.item()}")