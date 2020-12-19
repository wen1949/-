import cv2
import smtplib
from email.header import Header
import time
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('face_trainner/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX

idnum = 0

names = ['0','zhong','2','3','4']  #显示匹配到的用户姓名

cam = cv2.VideoCapture(0)     #调用笔记本摄像头

minW = 0.1*cam.get(3)          #摄像框大小
minH = 0.1*cam.get(4)

while True:              #循环保持
    ret, img = cam.read()                 #读取摄像头的内容
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)      #转灰色调

    #人脸识别
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH))
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        idnum, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        if confidence  < 60:                                         #如果相似值小于一百
            idnum = names[idnum]                                     #输出显示相似的用户名
            confidence = "{0}%".format(round(100 - confidence))
        else:
            idnum = "unknown"
            confidence = "{0}%".format(round(100 - confidence))

            if idnum == "unknown":
                #拍摄一张照片并储存，以便发送到邮箱
                print("数据库人脸无匹配....正在发送邮件告知....")
                now = time.strftime("%Y-%m-%d-%H_%M_%S")
                cv2.imwrite('img/img'+ now +'.jpg', img)

                # 第三方 SMTP 服务
                msg = MIMEMultipart('related')
                msg_str = """
            <p>发现未知人员....</p>
            <img src="cid:123">
            """
                msg.attach(MIMEText(msg_str, 'html', 'utf-8'))

                with open('img/img'+ now +'.jpg', 'rb') as f:
                    pic = MIMEImage(f.read())
                    pic.add_header('Content-ID', '<123>')
                    msg.attach(pic)

                    msg['From'] = '1967738236@qq.com'
                    msg['To'] = '1967738236@qq.com'
                    msg['Subject'] = Header("树莓派人脸入侵检测系统",'utf-8').encode()
                    smtp = smtplib.SMTP_SSL('smtp.qq.com')
                    smtp.login('1967738236@qq.com', 'moveuwnaroyhecca')
                    smtp.sendmail('1967738236@qq.com', '1967738236@qq.com', msg.as_string())

                    print("邮件发送成功")
                    time.sleep(5)

        cv2.putText(img, str(idnum), (x+5, y-5), font, 1, (0, 0, 255), 1)            #输出显示相似的用户名
        cv2.putText(img, str(confidence), (x+5, y+h-5), font, 1, (0, 0, 0), 1)         #输出相似度

    cv2.imshow('camera', img)     #保持画面

    k = cv2.waitKey(10)     #按esc退出
    if k == 27:
        break

cam.release()         #关闭摄像头
cv2.destroyAllWindows()