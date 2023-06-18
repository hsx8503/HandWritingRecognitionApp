import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from model import LeNet
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = LeNet().to(device)
state_dict = torch.load('./Lenet-new.pth', map_location=torch.device(device))

pretrained_dict = state_dict.copy()

# 转换预训练模型参数的数据类型为CUDA张量
for key in pretrained_dict:
    pretrained_dict[key] = pretrained_dict[key].to(device)


if __name__ == '__main__':
    net.eval()
    # 读取图像文件
    image = Image.open('./imgs/8.png').convert('L')
    image.show()

    # 定义图像变换
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 图像预处理
    image = transform(image)
    image = image.unsqueeze(0)  # 添加批次维度

    with torch.no_grad():
        image = image.to(device)
        output = net(image)

    # 处理模型的输出
    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)

    # 打印预测结果
    print(f"Predicted class: {predicted_class.item()}")