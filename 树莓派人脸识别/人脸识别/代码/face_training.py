import numpy as np
from PIL import Image
import os
import cv2

path = 'Facedata'   # 人脸数据路径

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]     # join函数的作用？--可能是转换数据吧！
    faceSamples = []
    ids = []

    #批量处理图库信息
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')   # 将图库图片转换为灰度
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])  #用户id

        # 人脸检测
        faces = detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x: x + w])
            ids.append(id)         #添加到列表ids中
    return faceSamples, ids        #返回的数值

print('训练面孔,这需要几秒钟，等待……')
faces, ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

recognizer.write(r'face_trainner\trainer.yml')    #保存返回的数据并命名文件
print( '\n '"{0} 面部训练完成，已退出程序".format(len(np.unique(ids))))
