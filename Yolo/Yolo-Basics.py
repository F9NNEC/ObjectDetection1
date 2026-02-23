from ultralytics import YOLO
import cv2

model = YOLO('../YOLO-Weights/yolov8n.pt')
results = model('images/3.png', show=True)

annotated = results[0].plot()

cv2.imshow("Detections", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()