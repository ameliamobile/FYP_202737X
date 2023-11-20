# """Region Counter and Unattended Bag Alert Code"""
# # TODO: To implement Object detection and Region counter in a specified region of interest on a video stream
# # TODO: Baggage Tracking:
# #  Implement object tracking to monitor the movement of unattended baggage.
# #  Using the centroid of the detected object to calculate the distance between person and chair
# #  To determine whether the suitcase is left unattended
# #  Trigger alerts if baggage remains unattended for an extended period.
#
# # TODO: For demo, you can display a video of people moving along the airport and to count the people crossing a line or
# #  something and to detect bags that are left unattended if possible.
# import os
# import cv2
# import math
# import time
# import numpy as np
# from shapely.geometry import Polygon
# from shapely.geometry.point import Point
#
# from ultralytics import YOLO
# from ultralytics.utils.plotting import Annotator, colors
#
# from collections import defaultdict
# track_history = defaultdict(list)
#
# current_region = None
# counting_regions = [
#     {
#         # TODO: Creating a rectangle region with polygon boundaries
#         'name': 'YOLOv8 Rectangle Region',
#         'polygon': Polygon([(0, 0), (640, 0), (640, 480), (0, 480)]),  #The whole frame
#         #'polygon': Polygon([(0, 0), (150, 0), (150, 650), (0, 650)]),  #left bounding region
#         #'polygon': Polygon([(50, 50), (590, 50), (590, 430), (50, 430)]),  #50 px offset of the frame
#         'person_counts': 0,
#         'dragging': False,
#         'region_color': (37, 255, 225),  # BGR Value
#         'text_color': (0, 0, 0),  # Region Text Color
#     },
#     ]
#
# def save_image(image, folder, filename):
#     if not os.path.exists(folder):
#         os.makedirs(folder)
#     cv2.imwrite(os.path.join(folder, filename), image)
#
# def demo_detection():
#     vid_frame_count = 0
#     cx1 = 150
#     person_crossed_x_axis = False
#
#
#     # TODO: Video Setup
#     # video_capture = path_x
#     # cap = cv2.VideoCapture(video_capture)
#     cap = cv2.VideoCapture('https://www.shutterstock.com/shutterstock/videos/1012107992/preview/stock-footage-london-'
#                            'england-may-sitting-with-a-coffee-noticing-an-apparently-unattended-bag-at-st.webm')
#     frame_width, frame_height = int(cap.get(5)), int(cap.get(6))
#
#     # TODO: Setup Model
#     model = YOLO(" ../YOLO-Weights/yolov8n.pt")
#
#     # TODO: Iterate over video frames
#     while cap.isOpened():
#         success, frame = cap.read()
#         if not success:
#             break
#         vid_frame_count += 1
#
#         # TODO: Extract the results
#         results = model.track(frame, persist=True)
#
#         if results[0].boxes.id is not None:
#             boxes = results[0].boxes.xywh.cpu()
#             track_ids = results[0].boxes.id.int().cpu().tolist()
#             clss = results[0].boxes.cls.cpu().tolist()
#             names = results[0].names
#
#             annotator = Annotator(frame, line_width=2, example=str(names))
#             cv2.line(frame, (cx1, 0), (cx1, 500), (255, 255, 255), 2)
#
#             crossed_track_ids = set()
#
#             for box, track_id, cls in zip(boxes, track_ids, clss):
#                 # TODO: Filter person and backpack results
#                 if names[cls] not in ['person', 'backpack']:
#                     continue
#
#                 x, y, w, h = box
#                 class_names = str(names[cls])
#
#                 xyxy = (x - w / 2), (y - h / 2), (x + w / 2), (y + h / 2)
#
#                 # TODO: Bounding box plot
#                 bbox_color = colors(cls, True)
#                 annotator.box_label(xyxy, color=bbox_color)
#
#                 # TODO: Check for detection inside region
#                 for region in counting_regions:
#                     if region['polygon'].contains(Point((x, y))):
#                         if class_names == 'person':
#                             # TODO: Calculate person bounding box centroid coordinates
#                             centroid_x_person = (x + (x + w)) / 2
#                             centroid_y_person = (y + (y + h)) / 2
#
#                             # Add this block to check if the person crosses the x-axis
#                             if centroid_x_person < cx1 and track_id not in crossed_track_ids:
#                                 region['person_counts'] += 1
#                                 crossed_track_ids.add(track_id)
#
#                         if class_names == 'backpack':
#                             # TODO: Calculate chair bounding box centroid coordinates
#                             centroid_x_backpack = (x + (x + w)) / 2
#                             centroid_y_backpack = (y + (y + h)) / 2
#
#                             # TODO: Calculate the distance between person and chair centroids
#                             distance = math.sqrt(
#                                 (centroid_x_person - centroid_x_backpack) ** 2 + (
#                                             centroid_y_person - centroid_y_backpack) ** 2)
#
#                             # TODO: Check if the bag is unattended (centroid distance > 100 pixels)
#                             if distance > 150:
#                                 # TODO: Record the time when bag is first detected unattended
#                                 if 'unattended_bag_time' not in region or region['unattended_bag_time'] is None:
#                                     region[
#                                         'unattended_bag_time'] = time.time()
#
#                                 # TODO: Calculate elapsed time since bag is unattended
#                                 elapsed_time = time.time() - region['unattended_bag_time']
#
#                                 # TODO: Display elapsed time in the bounding box label
#                                 label_with_time = f'Unattended: {elapsed_time:.2f} seconds'
#                                 annotator.box_label(xyxy, label_with_time, (0, 0, 255))
#
#                                 # TODO: Trigger alerts, actions, or notifications based on elapsed time if needed
#                                 threshold_time = 5.0  # Define your threshold time for triggering actions
#                                 if elapsed_time > threshold_time:
#                                     # Perform actions or trigger alerts based on the elapsed time
#                                     cv2.putText(frame, f'ABANDONED BAG ALERT!!', (300, 60), cv2.FONT_HERSHEY_SIMPLEX,
#                                                 0.7, (0, 0, 255), 2)  # Alert
#
#                                     # TODO: Capture and save image of the unattended bag
#                                     # Check if the image is already captured
#                                     if not region['image_captured']:
#                                         # Capture and save image of the unattended bag
#                                         bag_image = frame[int(y - h / 2):int(y + h / 2), int(
#                                             x - w / 2):int(x + w / 2)].copy()
#                                         save_image(bag_image, "unattended_bag_images",
#                                                    f'unattended_bag_{time.time()}.jpg')
#                                         # Set the image captured flag to True
#                                         region['image_captured'] = True
#                                     else:
#                                         print("Image already captured")
#
#                             else:
#                                 # Reset the unattended_bag_time if the distance is less than 100 pixels
#                                 region['unattended_bag_time'] = None
#
#                                 # Reset the image captured flag to False
#                                 region['image_captured'] = False
#
#
#         # TODO: Draw regions (Polygons/Rectangles)
#         for region in counting_regions:
#             region_label_person = str(region['person_counts'])
#             region_color = region['region_color']
#             region_text_color = region['text_color']
#
#             polygon_coords = np.array(region['polygon'].exterior.coords, dtype=np.int32)
#
#             cv2.putText(frame, f'Person Leaving: {region_label_person}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
#                         region_text_color, 2)
#             # TODO: To display the region area for counting
#             cv2.polylines(frame, [polygon_coords], isClosed=True, color=region_color, thickness=2)
#
#
#         # TODO: Reinitialize count for each region
#         for region in counting_regions:
#             region['person_counts'] = 0
#
#         yield frame
#     cv2.destroyAllWindows()
#
#



from ultralytics import YOLO
import cv2
import math
from sort import *
import cvzone

# TODO: Function that does the Object detection using YOLOv8 model on the UnitV2 Camera stream
def demo_detection():
    # TODO: Extract the video stream from the Ip address
    cap = cv2.VideoCapture("static/uploads/cars.mp4")  # For Video
    #cap = cv2.VideoCapture('https://www.shutterstock.com/shutterstock/videos/1012107992/preview/stock-footage-london-england-may-sitting-with-a-coffee-noticing-an-apparently-unattended-bag-at-st.webm')
    #cap.set(cv2.CAP_PROP_FPS, 30)  # Desired frame rate
    #frame_width, frame_height = int(cap.get(3)), int(cap.get(4))

    model = YOLO("../Yolo-Weights/yolov8l.pt")

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

    mask = cv2.imread("static/uploads/mask.png")

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
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
            cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                               scale=2, thickness=3, offset=10)

            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
                if totalCount.count(id) == 0:
                    totalCount.append(id)
                    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

        cvzone.putTextRect(img, f' Car Count: {len(totalCount)}', (50, 50))
        #cv2.putText(img, str(len(totalCount)), (255, 100),cv2.FONT_HERSHEY_PLAIN,5,(50,50,255),8)

        yield img
    # cv2.destroyAllWindows()