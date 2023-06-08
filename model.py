import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import torchvision.transforms as transforms
import os
from torch.autograd import Variable
from PIL import Image, ImageOps


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5,padding=2)
        self.Sigmoid = nn.Sigmoid()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=16,out_channels=120,kernel_size=5)

        self.fc1 = nn.Flatten()
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.Sigmoid(self.conv1(x))
        x = self.pool1(x)
        x = self.Sigmoid(self.conv2(x))
        x = self.pool2(x)
        x = self.conv3(x)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

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
def get_net():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LeNet()

    state_dict = torch.load('Lenet3.pth', map_location=torch.device(device))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# 对图像的初步处理
def get_num(img):
    # 转换为灰度图像，对大小进行缩放，生成二值图像，
    img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img_gray_resize = cv.resize(img_gray, (600, 600))
    ret, img_bw = cv.threshold(img_gray_resize, 200, 255, cv.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)

    # 实现对图中最小连通域的剔除，表现为净化页面效果
    img_open = cv.dilate(img_bw, kernel, iterations=2)
    num_labels, labels, stats, centroids = \
        cv.connectedComponentsWithStats(img_open, connectivity=8, ltype=None)
    for sta in stats:
        if sta[4] < 1000:
            cv.rectangle(img_open, tuple(sta[0:2]), tuple(sta[0:2] + sta[2:4]), (0, 0, 255), thickness=-1)
    return img_open


def get_roi(img_bw):
    # 粗略提取图片的文本区域
    img_bw_c = img_bw.sum(axis=1) / 255
    img_bw_r = img_bw.sum(axis=0) / 255
    all_sum = img_bw_c.sum(axis=0)
    # 对文本区域进行裁剪和补边，产生较为规整的方形区域
    if all_sum != 0:
        r_ind, c_ind = [], []
        # 过滤存在文本的区域
        for k, r in enumerate(img_bw_r):
            if r >= 5:
                r_ind.append(k)
        for k, c in enumerate(img_bw_c):
            if c >= 5:
                c_ind.append(k)
        if len(r_ind) == 0 or len(c_ind) == 0:
            return img_bw
        # 切出中间部分
        img_bw_sg = img_bw[c_ind[0]:c_ind[-1], r_ind[0]:r_ind[-1]]
        # 将图片缩到28x28的像素大小方便
        new_w, new_h = (28, 28)
        aspect_ratio = img_bw_sg.shape[1] / img_bw_sg.shape[0]
        if new_w / new_h > aspect_ratio:
            new_w = int(new_h * aspect_ratio)
        else:
            new_h = int(new_w / aspect_ratio)
        img_bw_sg = cv.resize(img_bw_sg, (new_w, new_h))
        w_padding = int((28 - new_w) / 2)
        h_padding = int((28 - new_h) / 2)
        img_bw_sg_bord = cv.copyMakeBorder(img_bw_sg, h_padding, h_padding, w_padding, w_padding, cv.BORDER_CONSTANT,
                                           value=[0, 0, 0])
        return img_bw_sg_bord
    else:
        return img_bw


def predict(img):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_invert = Image.fromarray(img)
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image_tensor = Variable(transform(image_invert).unsqueeze(0).float()).to(device)
    model = get_net()
    with torch.no_grad():
        pred = model(image_tensor)
        predicted = classes[torch.argmax(pred[0])]
        result = predicted
    return result
