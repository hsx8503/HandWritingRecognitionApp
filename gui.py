import cv2 as cv
import numpy as np
import os
from time import time
from model import get_net, get_num, get_roi, predict

capture = cv.VideoCapture(0,cv.CAP_DSHOW)
capture.set(3, 1920)
capture.set(4, 1080)
model = get_net()

while (True):
    ret, frame = capture.read()
    since = time()
    if ret:
        img_bw = get_num(frame)
        img_bw_sg = get_roi(img_bw)
        # 等待用户在键盘输入，除了ESC以外的输入，以启动识别图片
        cv.imshow("img",img_bw_sg)
        c = cv.waitKey(1) & 0xff
        if c == 27:
            capture.release()
            break
        result = predict(img_bw_sg)

        img_show = cv.resize(frame, (600, 600))
        end_predict = time()
        fps = round(1/(end_predict - since))
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(img_show, "The number is:" + str(result), (1, 30), font, 1, (0, 0, 255), 2)
        cv.putText(img_show, "FPS:" + str(fps), (1, 90), font, 1, (255, 0, 0), 2)
        cv.imshow("result", img_show)
        cv.waitKey(1)
        print(result)
        print("*" * 50)
        print("The number is:", result)

    else:
        print("请检查摄像头！！")
        break

























