from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')  #Pre-trained model
results = model('../static/uploads/bus.jpg', show=True)  # Insert image for object detection

cv2.waitKey(0)

