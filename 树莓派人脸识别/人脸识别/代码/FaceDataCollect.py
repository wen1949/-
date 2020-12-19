#coding=utf-8

import cv2
import os

cap = cv2.VideoCapture(0)  # 调用笔记本内置摄像头，所以参数为0，如果有其他的摄像头可以调整参数为1，2

face_detector = cv2.CascadeClassifier(r'C:\pytho3\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')

face_id = input('\n 输入用户id:')

print('\n 初始化面临捕获，请看着摄像机等着……')

count = 0

while True:

    sucess, img = cap.read()   # 从摄像头读取图片

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # 转为灰度图片

    # 检测人脸
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+w), (255, 0, 0))
        count += 1     #计算拍摄的张数

        cv2.imwrite("Facedata/User." + str(face_id) + '.' + str(count) + '.jpg', gray[y: y + h, x: x + w])   #保存每张图片的命名格式

        cv2.imshow('image', img)   # 保持画面的持续。

    k = cv2.waitKey(1)      # 通过esc键退出摄像
    if k == 27:
        break

    elif count >= 1000:  # 得到1000个样本后退出摄像
        break

cap.release()  # 关闭摄像头
cv2.destroyAllWindows()