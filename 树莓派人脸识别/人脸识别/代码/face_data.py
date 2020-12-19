#coding=utf-8
import cv2
import numpy as np
import  os

#检测人脸
def detect_face(img):
    gray =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  #转灰调

    #加载opencv的人脸你分类器
    face_cascade =cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    face = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,       #检测多尺度图像
    )
    if (len(face)==0):        #如果未检测到人脸，则返回原始图像
        return None,None

    (x,y,w,h) =face[0]      #只有一张脸，xy为左上角坐标，wh为矩形的宽高
    return gray[y:y+w,x:x+h],face[0]      #返回正面部分

#该函数将读取所有的训练图像，从每个图像检测人脸并返回两个相同大小的列表，分别为脸部信息和标签
def prepare_training_data(data_folder_path):
    dirs =os.listdir(data_folder_path)  #获取文件夹中的目录

    faces =[]
    labels =[]

    for dir_name in dirs:         #浏览每个目录并访问其中的图像
        labels = int(dir_name)

                #建立包含当前主题的图像目录路径
        subject_dir_path =data_folder_path + "/" + dir_name

        subject_images_name = os.listdir(subject_dir_path)  #获取给定主题目录内的图像名称

        for image_name in subject_images_name:
            image_path = subject_images_name + "/" +image_name    #建立图像路径
            image = cv2.imread(image_path)        #读取图片

            cv2.imshow("训练图像中......", image)
            cv2.waitKey(100)

            #人脸检测
            face, label,rect =detect_face(image)
            if face is not None:           #如果未检出人脸
                faces.append(face)          #将脸添加到脸部列表并添加相应的标签
                labels.append(label)

                cv2.waitKey(1)
                cv2.destroyAllWindows()
                return faces,labels         #最终返回人脸与标签

