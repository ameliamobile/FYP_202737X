from ultralytics import YOLO
import cv2
import math

def unit_detection():
    # extract the video stream from the Ip address
    cap = cv2.VideoCapture('http://10.254.239.1/video_feed')
    cap.set(cv2.CAP_PROP_FPS, 30)  # Desired frame rate
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    model = YOLO(" ../YOLO-Weights/yolov8n.pt")

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

    while True:
        success, img = cap. read()
        # Doing detection using YOLOv8 frame by frame
        # Stream = True will use the generator and it is more efficient than normal
        results = model(img, stream=True)
        # Once we have the results, we can check for the individual bounding boxes and see how well it performs
        # Once we have the results, we will loop through them and we will have the bounding boxes for each of the results
        # We will loop through each of the bounding box
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Confidence score
                # print(box.conf[0])
                conf = math.ceil((box.conf[0] * 100))  # Calculates the confidence score and rounds the confidence

                # Class Name
                cls = int(box.cls[0])
                class_name = classNames[cls]

                # Bounding box
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
                cv2.putText(img, label, (x1, y1 - 2), 0, 0.7, [255, 255, 255], thickness=2, lineType=cv2.LINE_AA)  # Outline



        yield img
    cv2.destroyAllWindows()