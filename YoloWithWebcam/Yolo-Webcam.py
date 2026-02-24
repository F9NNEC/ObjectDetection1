from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture(0) #0 jika hanya ada 1 webcam, else 1
cap.set(3, 1280) #width
cap.set(4, 720) #height

model = YOLO('../YOLO-Weights/yolov8n.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush", "glasses"
              ]

while True:
    succees, img = cap.read()
    result = model(img, stream=True)
    for r in result:
        boxes = r.boxes
        for box in boxes:
            # bounding box
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            # print(x1,y1,x2,y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),3)

            # cvzone
            # x1,y1,w,h = box.xywh[0]
            # bbox = int(x1),int(y1),int(w),int(h)
            # cvzone.cornerRect(img,bbox)

            w,h = x2-x1,y2-y1 # widht,height
            cvzone.cornerRect(img,(x1,y1,w,h), colorR=(255,17,0), colorC=(252,240,3))
            print(x1,y1,w,h)

            # confidence
            confidence = (math.ceil(box.conf[0]*100))/100 # 0-1
            print(confidence)

            # classify Name
            cls = int(box.cls[0])

            cvzone.putTextRect(img, f'{classNames[cls]} {confidence}',(max(0, x1), max(50, y1-20)), scale=0.9, thickness=1, colorR=(255,17,0))

    cv2.imshow("image", img)
    cv2.waitKey(1)