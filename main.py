import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
import glob
import shutil as sh
import os


sh.rmtree(".\\open") #xóa folder open và close trc đó để test lại từ đầu
sh.rmtree(".\\close")
os.mkdir("open") #xóa rồi tạo mới
os.mkdir("close")

screenW=640 #chiều ngang
screenH=360 #chiều dọc

cap = cv2.VideoCapture('Video3.mp4')
detector = FaceMeshDetector(maxFaces=1)
plotY = LivePlot(screenW, screenH, [14, 54], invert=True)

idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 221, 222, 223, 224, 225, 243, 70, 63, 105, 66, 107]
ratioList = []
counterList = [] #danh sách kiểu (0,1,0,1,...) để xác định việc nhắm mở
blinkCounter = 0 # điều kiện tính blink: mở rồi nhắm rồi mở, so phần tử cuối counterList với 2 cái trc đó

color = (255, 0, 255) #màu để plot

while True:

    #if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        #cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    #chạy cái trên để video loop vô tận

    success, img = cap.read()
    img, faces = detector.findFaceMesh(img, draw=False)
    eye_weight=0 #để xem nhìn thấy bao nhiêu phần con mắt

    if faces:
        face = faces[0]
        for id in idList:
            cv2.circle(img, face[id], 5, color, cv2.FILLED)
            if face[id][0]>0 and face[id][1]>0: #điểm id này nằm trong màn hình
                eye_weight+=1 #22 nghĩa là thấy toàn bộ con mắt
        #print(eye_weight)

        leftUp = face[159]
        leftDown = face[23]
        leftLeft = face[130]
        leftRight = face[243]

        lenghtVer, _ = detector.findDistance(leftUp, leftDown)
        lenghtHor, _ = detector.findDistance(leftLeft, leftRight)

        #cv2.line(img, leftUp, leftDown, (0, 200, 0), 3)
        #cv2.line(img, leftLeft, leftRight, (0, 200, 0), 3)

        ratio = (lenghtVer / lenghtHor) * 100 #tỉ lệ chiều ngang và dọc của con mắt
        ratioList.append(ratio)
        if len(ratioList) > 3: #smooth lại theo step = 3
            ratioList.pop(0)
        ratioAvg = sum(ratioList) / len(ratioList)
        print(ratioAvg)
        if cap.get(cv2.CAP_PROP_POS_FRAMES)>=5 and eye_weight >=20 : #Nếu thấy đủ con mắt và bắt đầu từ frame 5 trở đi
            if ratio <= 34:
                cv2.imwrite(".\\close\\%d.jpg" % (len(glob.glob(".\\close\\*.jpg"))), img) #save vào close với số thứ tự
            else:
                cv2.imwrite(".\\open\\%d.jpg" % (len(glob.glob(".\\open\\*.jpg"))), img) #save vào open với số thứ tự

            if ratioAvg <= 34: #plot theo màu
                color = (0, 200, 0)
            else:
                color = (255, 0, 255)

            if ratioAvg <= 34 and (len(counterList)==0 or counterList[-1] == 1): #nếu mắt mở sau khi mở và nhắm trước đó -> blink
                counterList.append(0)
            if ratioAvg > 34 and (len(counterList)==0 or counterList[-1] == 0):
                counterList.append(1)
                if len(counterList)>=3:
                    blinkCounter+=1


        else:
            color = (0, 200, 0)

        cvzone.putTextRect(img, f'{blinkCounter}', (50, 100),
                           colorR=color) #f'Blink Count: {blinkCounter}'

        imgPlot = plotY.update(ratioAvg, color) #ratioAvg for average by step of 3
        img = cv2.resize(img, (screenW, screenH))
        imgStack = cvzone.stackImages([img, imgPlot], 2, 1)
    else:
        img = cv2.resize(img, (screenW, screenH))
        imgStack = cvzone.stackImages([img, img], 2, 1)

    cv2.imshow("Image", imgStack)
    cv2.waitKey(10)