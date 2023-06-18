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
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)#定义一个二维卷积层，并将该层赋值给类属性conv1。输入通道数为1，输出通道数为6，卷积核大小为5*5，padding为2
        self.Sigmoid = nn.Sigmoid()#定义一个Sigmoid激活函数，并将该函数赋值给类属性Sigmoid。
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)#定义一个平均池化层,使用平均池化的方式，池化核大小为2*2，步长为2
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)#定义一个二维卷积层
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)#定义一个平均池化层
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)#定义一个二维卷积层

        self.fc1 = nn.Flatten()#定义一个展平层，用于将数据展开成一维张量,并将该层赋值给类属性fc1。
        self.fc2 = nn.Linear(120, 84)#定义一个线性变换层,输入大小为120，输出大小为84，
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):#定义前向传播函数，其中x是输入数据。即网络结构中数据从输入到输出的流程
        x = self.Sigmoid(self.conv1(x))# 对输入进行第一次卷积和Sigmoid激活操作。得到特征图
        x = self.pool1(x)# 对输入进行第一次卷积和Sigmoid激活操作。
        x = self.Sigmoid(self.conv2(x))#对上一步池化的结果进行第二次卷积和Sigmoid激活操作。
        x = self.pool2(x)
        x = self.conv3(x)#对上一步池化的结果进行第三次卷积操作。

        x = self.fc1(x)#对上一步卷积的结果进行展平操作
        x = self.fc2(x)#对上一步展平的结果进行线性变换操作。
        x = self.fc3(x)#对上一步线性变换的结果进行线性变换操作。
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
    net = LeNet().to(device)
    state_dict = torch.load('Lenet3.pth', map_location=torch.device(device))
    net.load_state_dict(state_dict)
    net.eval()
    print("get_net函数节点3")

    print("get_net函数节点4")

    print("get_net函数节点5")
    return net
'''
def get_net1():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'#根据计算机是否有可用的 CUDA 设备来选择使用 "cuda" 或 "cpu"。
    model = LeNet()#创建一个 LeNet 模型对象。
    #使用 torch.load 函数加载名为 "Lenet3.pth" 的模型参数，并将其映射到设备上
    state_dict = torch.load('Lenet3.pth', map_location=torch.device(device))
    model.load_state_dict(state_dict)#将加载的模型参数应用于 LeNet 模型对象。
    model.to(device)#将模型对象移动到指定的设备上。
    #model.eval()#将模型对象设置为评估模式。
    return model#返回加载好的、可用于推理的 LeNet 模型对象。

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

#预处理数字图像以便可以输入到机器学习模型中进行识别
def preprocess_img(img):
    # 将输入图像转换为灰度图像
    img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    #缩放图像大小为28x28像素
    img_resize = cv.resize(img_gray, (28, 28))
    #反转图像颜色
    img_invert = cv.bitwise_not(img_resize)
    #将图像像素值归一化为0到1之间的浮点数
    img_norm = img_invert / 255.0
    return img_norm
def preprocess_img1(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # convert to grayscale
    blurred = cv.GaussianBlur(gray, (5, 5), 0) # apply Gaussian blur
    thresh = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 11, 2) # apply adaptive thresholding
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3)) # define a 3x3 rectangular kernel
    eroded = cv.erode(thresh, kernel, iterations=1) # apply erosion
    dilated = cv.dilate(eroded, kernel, iterations=1) # apply dilation
    return dilated
def preprocess_and_extract_text(img):
    # Preprocess the image using the given code
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # convert to grayscale
    blurred = cv.GaussianBlur(gray, (5, 5), 0) # apply Gaussian blur
    thresh = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 11, 2) # apply adaptive thresholding
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3)) # define a 3x3 rectangular kernel
    eroded = cv.erode(thresh, kernel, iterations=1) # apply erosion
    dilated = cv.dilate(eroded, kernel, iterations=1) # apply dilation

    # Find contours in the preprocessed image
    contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Extract the bounding box of the largest contour (presumably the text)
    max_contour = max(contours, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(max_contour)

    # Add some padding around the bounding box
    padding = 8
    x -= padding
    y -= padding
    w += 2 * padding
    h += 2 * padding

    # Make sure the new bounding box doesn't exceed the image boundaries
    x = max(0, x)
    y = max(0, y)
    w = min(w, img.shape[1] - x)
    h = min(h, img.shape[0] - y)

    # Extract the ROI containing the text with padding and resize it to 28x28
    roi = dilated[y:y + h, x:x + w]
    resized_roi = cv.resize(roi, (28, 28))

    return resized_roi
'''
def preprocess_and_extract_text1(img):
    # Preprocess the image using the given code
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # convert to grayscale
    blurred = cv.GaussianBlur(gray, (5, 5), 0) # apply Gaussian blur
    thresh = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 11, 2) # apply adaptive thresholding
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3)) # define a 3x3 rectangular kernel
    eroded = cv.erode(thresh, kernel, iterations=1) # apply erosion
    dilated = cv.dilate(eroded, kernel, iterations=1) # apply dilation

    # Find contours in the preprocessed image
    contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Extract the bounding box of the largest contour (presumably the text)
    max_contour = max(contours, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(max_contour)

    # Calculate aspect ratio of the bounding box
    aspect_ratio = w / h

    # Set target height and width for resizing while preserving aspect ratio
    target_size = 28
    if aspect_ratio > 1:
        target_width = target_size
        target_height = int(target_size / aspect_ratio)
    else:
        target_height = target_size
        target_width = int(target_size * aspect_ratio)

    # Resize the ROI containing the text while preserving aspect ratio
    roi = dilated[y:y + h, x:x + w]
    resized_roi = cv.resize(roi, (target_width, target_height), interpolation=cv.INTER_AREA)

    # Add padding to fit the resized image into a 28x28 box (with black color)
    padded_roi = np.zeros((target_size, target_size), dtype=np.uint8)
    pad_w = int((target_size - target_width) / 2)
    pad_h = int((target_size - target_height) / 2)
    padded_roi[pad_h:pad_h+target_height, pad_w:pad_w+target_width] = resized_roi

    return padded_roi
# def get_roi(img_bw):
#     # 粗略提取图片的文本区域
#     img_bw_c = img_bw.sum(axis=1) / 255
#     img_bw_r = img_bw.sum(axis=0) / 255
#     all_sum = img_bw_c.sum(axis=0)
#     # 对文本区域进行裁剪和补边，产生较为规整的方形区域
#     if all_sum != 0:
#         r_ind, c_ind = [], []
#         # 过滤存在文本的区域
#         for k, r in enumerate(img_bw_r):
#             if r >= 5:
#                 r_ind.append(k)
#         for k, c in enumerate(img_bw_c):
#             if c >= 5:
#                 c_ind.append(k)
#         if len(r_ind) == 0 or len(c_ind) == 0:
#             return img_bw
#         # 切出中间部分
#         img_bw_sg = img_bw[c_ind[0]:c_ind[-1], r_ind[0]:r_ind[-1]]
#         # 将图片缩到28x28的像素大小方便
#         new_w, new_h = (28, 28)
#         aspect_ratio = img_bw_sg.shape[1] / img_bw_sg.shape[0]
#         if new_w / new_h > aspect_ratio:
#             new_w = int(new_h * aspect_ratio)
#         else:
#             new_h = int(new_w / aspect_ratio)
#         img_bw_sg = cv.resize(img_bw_sg, (new_w, new_h))
#         w_padding = int((28 - new_w) / 2)
#         h_padding = int((28 - new_h) / 2)
#         img_bw_sg_bord = cv.copyMakeBorder(img_bw_sg, h_padding, h_padding, w_padding, w_padding, cv.BORDER_CONSTANT,
#                                            value=[0, 0, 0])
#         return img_bw_sg_bord
#     else:
#         return img_bw

# def get_roi1(img_bw):
#     # 粗略提取图片的文本区域
#     img_bw_c = img_bw.sum(axis=1) / 255#对输入的二值化图像进行纵向求和，并将其除以255，得到每行上白色像素点的数量。
#     img_bw_r = img_bw.sum(axis=0) / 255#对输入的二值化图像进行横向求和，并将其除以255，得到每列上白色像素点的数量。
#     all_sum = img_bw_c.sum(axis=0)#计算所有行中白色像素点的总数。
#
#     # 对文本区域进行裁剪和补边，产生较为规整的方形区域
#     if all_sum != 0:
#         r_ind, c_ind = [], []
#         # 过滤存在文本的区域
#         for k, r in enumerate(img_bw_r):#遍历横向白色像素点数量列表（即每列上白色像素点的数量），其中k是索引，r是值。
#             if r >= 5:#如果该列上白色像素点数量大于等于5个，则说明这列属于文本区域，将它的索引k添加到r_ind列表中。
#                 r_ind.append(k)
#         for k, c in enumerate(img_bw_c):
#             if c >= 5:
#                 c_ind.append(k)
#         if len(r_ind) == 0 or len(c_ind) == 0:#如果r_ind和c_ind列表中至少有一个为空，则说明无法确定文本区域的位置，返回原图像。
#             return img_bw
#
#         # 切出中间部分
#         img_bw_sg = img_bw[c_ind[0]:c_ind[-1], r_ind[0]:r_ind[-1]]#根据行列索引切割出文本区域的部分图像。
#
#         # 将图片缩到28x28的像素大小方便
#         new_w, new_h = (28, 28)
#         aspect_ratio = img_bw_sg.shape[1] / img_bw_sg.shape[0]#计算文本区域的横纵比。
#         if aspect_ratio > 1:
#             new_w = int(aspect_ratio * new_h)
#         else:
#             new_h = int(new_w / aspect_ratio)
#
#         img_bw_sg_resized = cv.resize(img_bw_sg, (new_w, new_h))#将文本区域缩放到新的大小。
#
#         w_padding = int((28 - new_w) / 2)#计算宽度方向上的填充量，使得新图像在28x28的边框中居中显示。
#         h_padding = int((28 - new_h) / 2)
#
#         img_bw_sg_padded = cv.copyMakeBorder(img_bw_sg_resized, h_padding, h_padding, w_padding, w_padding, cv.BORDER_CONSTANT,
#                                            value=[0, 0, 0])
#         return img_bw_sg_padded
#     else:
#         return img_bw

def predict(img):
    print("predict函数节点1")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_invert = Image.fromarray(img).convert('L')
    print("predict函数节点2")
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image_tensor = transforms.ToTensor()(image_invert)
    image_invert.show()
    model = get_net()
    print("predict函数节点6")
    image_tensor = Variable(torch.unsqueeze(image_tensor, dim=0).float()).to(device)
    print(image_tensor.shape)
    print(model.fc2.weight.shape)
    print("predict函数节点7")

    with torch.no_grad():
        pred = model(image_tensor)
        print("predict函数节点9")
        predicted = classes[torch.argmax(pred[0])]

    print("predict函数节点8")
    return predicted

# def predict1(img):
#     try:
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         image_invert = Image.fromarray(img)
#         transform = transforms.Compose([
#             transforms.Resize(32),
#             transforms.Resize((28, 28)),
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,))
#         ])
#         image_tensor = Variable(transform(image_invert).unsqueeze(0).float()).to(device)
#         model = get_net()
#         with torch.no_grad():
#             pred = model(image_tensor)
#             predicted = classes[torch.argmax(pred[0])]
#             result = predicted
#         return result
#     except Exception as e:
#         print(f"Error in predict: {e}")
#         return None