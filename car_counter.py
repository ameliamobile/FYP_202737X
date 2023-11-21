from ultralytics import YOLO
import cv2
import math
from sort import *
import cvzone

# TODO: Function that does the Object detection using YOLOv8 model on the UnitV2 Camera stream
def car_detection():
    # TODO: Extract the video stream from the Ip address
    cap = cv2.VideoCapture("static/uploads/cars.mp4")  # For Video

    model = YOLO("../Yolo-Weights/yolov8n.pt")

    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "potted lant", "bed",
                  "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"]

    mask = cv2.imread("static/uploads/carmask.png")

    # Tracking
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

    limits = [400, 297, 673, 297]
    totalCount = []

    while True:
        success, img = cap.read()
        imgRegion = cv2.bitwise_and(img, mask)
        # Doing detection using YOLOv8 frame by frame
        # Stream = True will use the generator and it is more efficient than normal
        results = model(imgRegion, stream=True)

        detections = np.empty((0, 5))
        # Once we have the results, we can check for the individual bounding boxes and see how well it performs
        # We will loop through them, and we will have the bounding boxes for each of the results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # TODO: Confidence score
                # print(box.conf[0])
                conf = math.ceil((box.conf[0] * 100))  # Calculates the confidence score and rounds the confidence

                # TODO: Class Name
                cls = int(box.cls[0])
                class_name = classNames[cls]

                if class_name == "car" and conf > 0.3:
                    # TODO: Bounding box
                    x1, y1, x2, y2 = box.xyxy[0]  # output is in tensors
                    # print(x1, y1, x2, y2)
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Converting output from tensors into integers
                    # print(x1, y1, x2, y2)

                    # cv2.rectangle(image, start_point, end_point, color, thickness)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

                    # Combining the label and confidence score
                    label = f'{class_name} {conf}%'
                    t_size = cv2.getTextSize(label, 0, fontScale=0.7, thickness=2)[0]  # Size of the label
                    # print(t_size)
                    c2 = x1 + t_size[0], y1 - t_size[1] - 3
                    cv2.rectangle(img, (x1, y1), c2, [0, 255, 0], -1, cv2.LINE_AA)  # Filled
                    cv2.putText(img, label, (x1, y1 - 2), 0, 0.7, [255, 255, 255], thickness=2,
                                lineType=cv2.LINE_AA)  # Outline

                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))

        resultsTracker = tracker.update(detections)

        cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

        for result in resultsTracker:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(result)
            w, h = x2 - x1, y2 - y1

            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
                if totalCount.count(id) == 0:
                    totalCount.append(id)
                    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

        cvzone.putTextRect(img, f' Car Count: {len(totalCount)}', (50, 50))

        yield img