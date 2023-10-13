from ultralytics import YOLO
import cv2
import math

def video_detection(path_x):
    video_capture = path_x
    # Create a Webcam object
    cap = cv2.VideoCapture(video_capture)  # Webcam
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    #out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

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
                x1, y1, x2, y2 = box.xyxy[0]  # output is in tensors
                #print(x1, y1, x2, y2)
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Converting output from tensors into integers
                print(x1, y1, x2, y2)
                # Creating the bounding boxes around the detective object
                # cv2.rectangle(image, start_point, end_point, color, thickness)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                # Confidence score value is in tensor
                #print(box.conf[0])
                conf = math.ceil((box.conf[0]*100))/100     # Calculates the confidence score and rounds the confidence
                # Extracts the class ID of the detected object
                cls = int(box.cls[0])
                class_name = classNames[cls]
                # Combining the label and confidence score
                label = f'{class_name}{conf}'
                # Positioning of the label and confidence score
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]  # Size of the label
                print(t_size)
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)  # filled
                cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)  # outline

        yield img
        # out.write(img)
        # cv2.imshow("image", img)
        # if cv2.waitKey(1) & 0xFF==ord('1'):
        # break
        # out.release()
    cv2.destroyAllWindows()