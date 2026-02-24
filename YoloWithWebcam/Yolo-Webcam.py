from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture(0) #0 jika hanya ada 1 webcam, else 1
cap.set(3, 1280) #width
cap.set(4, 720) #height

model = YOLO('../YOLO-Weights/yolov8n.pt')

while True:
    succees, img = cap.read()
    result = model(img, stream=True)
    for r in result:
        boxes = r.boxes
        for box in boxes:
            # opencv
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            # print(x1,y1,x2,y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),3)

            # cvzone
            # x1,y1,w,h = box.xywh[0]
            # bbox = int(x1),int(y1),int(w),int(h)
            # cvzone.cornerRect(img,bbox)

            w,h = x2-x1,y2-y1 # widht,height
            cvzone.cornerRect(img,(x1,y1,w,h))
            print(x1,y1,w,h)

            confidence = (math.ceil(box.conf[0]*100))/100 # 0-1
            print(confidence)

            cvzone.putTextRect(img, f'{confidence}',(max(0, x1), max(50, y1-20)))

    cv2.imshow("image", img)
    cv2.waitKey(1)