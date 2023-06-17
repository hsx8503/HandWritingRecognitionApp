import sys

import cv2
import cv2 as cv
import numpy as np
from PIL import Image

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QFileDialog, QMainWindow
from matplotlib import pyplot as plt

from GuiDesign import Ui_MainWindow
from model import predict, get_roi, get_num


class PyQtMainEntry(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.camera = cv.VideoCapture(0)
        self.is_camera_opened = False  # 摄像头有没有打开标记

        # 定时器：30ms捕获一帧
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.queryFrame)
        self.timer.setInterval(30)

    def openCamera(self):
        self.is_camera_opened = ~self.is_camera_opened
        if self.is_camera_opened:
            self.openCameraBtn.setText("关闭摄像头")
            self.timer.start()
        else:
            self.openCameraBtn.setText("打开摄像头")
            self.timer.stop()

    def capture(self):
        if not self.is_camera_opened:
            return
        self.captured = self.frame
        #
        rows, cols, channels = self.captured.shape
        bytesPerLine = channels * cols
        # Qt显示图片时，需要先转换成QImgage类型
        QImg = QImage(self.captured.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
        self.cameraImg.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.cameraImg.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def readImage(self):
        filename,  _ = QFileDialog.getOpenFileName(self, '打开图片')
        if filename:
            self.captured = cv.imread(str(filename))
            # OpenCV图像以BGR通道存储，显示时需要从BGR转到RGB
            self.captured = cv.cvtColor(self.captured, cv.COLOR_BGR2RGB)

            rows, cols, channels = self.captured.shape
            bytesPerLine = channels * cols
            QImg = QImage(self.captured.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
            self.forderImg.setPixmap(QPixmap.fromImage(QImg).scaled(
                self.forderImg.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    # def btnGray_Clicked(self):
    #     '''
    #     灰度化
    #     '''
    #     # 如果没有捕获图片，则不执行操作
    #     if not hasattr(self, "captured"):
    #         return
    #
    #     self.cpatured = cv.cvtColor(self.captured, cv.COLOR_RGB2GRAY)
    #
    #     rows, columns = self.cpatured.shape
    #     bytesPerLine = columns
    #     # 灰度图是单通道，所以需要用Format_Indexed8
    #     QImg = QImage(self.cpatured.data, columns, rows, bytesPerLine, QImage.Format_Indexed8)
    #     self.labelResult.setPixmap(QPixmap.fromImage(QImg).scaled(
    #         self.labelResult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
    #
    # def btnThreshold_Clicked(self):
    #     '''
    #     Otsu自动阈值分割
    #     '''
    #     if not hasattr(self, "captured"):
    #         return
    #
    #     _, self.cpatured = cv.threshold(
    #         self.cpatured, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    #
    #     rows, columns = self.cpatured.shape
    #     bytesPerLine = columns
    #     # 阈值分割图也是单通道，也需要用Format_Indexed8
    #     QImg = QImage(self.cpatured.data, columns, rows, bytesPerLine, QImage.Format_Indexed8)
    #     self.labelResult.setPixmap(QPixmap.fromImage(QImg).scaled(
    #         self.labelResult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    @QtCore.pyqtSlot()
    def queryFrame(self):
        ret, self.frame = self.camera.read()

        img_rows, img_cols, channels = self.frame.shape
        bytesPerLine = channels * img_cols

        cv.cvtColor(self.frame, cv.COLOR_BGR2RGB, self.frame)
        QImg = QImage(self.frame.data, img_cols, img_rows, bytesPerLine, QImage.Format_RGB888)
        self.cameraImg.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.cameraImg.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def predict_capture(self):
        if not hasattr(self, "captured"):
            return
        self.cpatured = cv.cvtColor(self.captured, cv.COLOR_RGB2GRAY)

        # print("节点2")
        # image_array = np.array(QImg.toNumpy())
        img_bw = get_num(self.cpatured)
        img_bw_sg = get_roi(img_bw)

        # convert to gray
        print("节点2")

        result = predict(img_bw_sg)
        print("节点3")

        self.result_capture.setText(str(result))


    def predict_folder(self):
        if not hasattr(self, "captured"):
            return
        self.cpatured = cv.cvtColor(self.captured, cv.COLOR_RGB2GRAY)

        # print("节点2")
        # image_array = np.array(QImg.toNumpy())
        # img_bw = get_num(image_array)
        # img_bw_sg = get_roi(img_bw)

        # convert to gray
        print("节点2")

        result = predict(self.cpatured)
        print("节点3")

        self.result_folder.setText(str(result))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = PyQtMainEntry()
    window.show()
    sys.exit(app.exec_())
