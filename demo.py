import os
import cv2
import math
import time
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.point import Point

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

from collections import defaultdict
track_history = defaultdict(list)

current_region = None
counting_regions = [
    {
        # TODO: Creating a rectangle region with polygon boundaries
        'name': 'YOLOv8 Rectangle Region',
        'polygon': Polygon([(0, 0), (1950, 0), (1950, 1300), (0, 1300)]),  #The whole frame
        'person_counts': 0,
        'chair_counts': 0,
        'dragging': False,
        'region_color': (37, 255, 225),  # BGR Value
        'text_color': (0, 0, 0),  # Region Text Color
    },
    ]

def save_image(image, folder, filename):
    if not os.path.exists(folder):
        os.makedirs(folder)
    cv2.imwrite(os.path.join(folder, filename), image)

def demobag_detection():
    vid_frame_count = 0

    # TODO: Video Setup
    cap = cv2.VideoCapture("static/uploads/suitcase.mp4")  # For Video

    # TODO: Setup Model
    model = YOLO(" ../YOLO-Weights/yolov8n.pt")

    # TODO: Iterate over video frames
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        vid_frame_count += 1

        # TODO: Extract the results
        results = model.track(frame, persist=True)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()
            names = results[0].names

            annotator = Annotator(frame, line_width=2, example=str(names))

            for box, track_id, cls in zip(boxes, track_ids, clss):
                # TODO: Filter person and chair results
                if names[cls] not in ['person', 'suitcase']:
                    continue

                x, y, w, h = box
                class_names = str(names[cls])

                xyxy = (x - w / 2), (y - h / 2), (x + w / 2), (y + h / 2)

                # TODO: Bounding box plot
                bbox_color = colors(cls, True)
                annotator.box_label(xyxy, class_names, color=bbox_color)

                # TODO: Tracking Lines plot
                track = track_history[track_id]
                track.append((float(x), float(y)))
                if len(track) > 30:
                    track.pop(0)
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=bbox_color, thickness=2)

                # TODO: Check for detection inside region
                for region in counting_regions:
                    if region['polygon'].contains(Point((x, y))):
                        if class_names == 'person':
                            # TODO: Calculate person bounding box centroid coordinates
                            centroid_x_person = (x + (x + w)) / 2
                            centroid_y_person = (y + (y + h)) / 2

                        elif class_names == 'suitcase':
                            # TODO: Calculate suitcase bounding box centroid coordinates
                            centroid_x_suitcase = (x + (x + w)) / 2
                            centroid_y_suitcase = (y + (y + h)) / 2

                            # TODO: Calculate the distance between person and chair centroids
                            distance = math.sqrt(
                                (centroid_x_person - centroid_x_suitcase) ** 2 + (
                                            centroid_y_person - centroid_y_suitcase) ** 2)

                            # TODO: Check if the bag is unattended (centroid distance > 200 pixels)
                            if distance > 200:
                                # TODO: Record the time when bag is first detected unattended
                                if 'unattended_bag_time' not in region or region['unattended_bag_time'] is None:
                                    region[
                                        'unattended_bag_time'] = time.time()

                                # TODO: Calculate elapsed time since bag is unattended
                                elapsed_time = time.time() - region['unattended_bag_time']

                                # TODO: Display elapsed time in the bounding box label
                                label_with_time = f'Unattended: {elapsed_time:.2f} seconds'
                                annotator.box_label(xyxy, label_with_time, (0, 0, 255))

                                # TODO: Trigger alerts, actions, or notifications based on elapsed time if needed
                                threshold_time = 15.0  # Define your threshold time for triggering actions
                                if elapsed_time > threshold_time:
                                    # Perform actions or trigger alerts based on the elapsed time
                                    cv2.putText(frame, f'ABANDONED BAG ALERT!!', (300, 200), cv2.FONT_HERSHEY_SIMPLEX,
                                                2, (0, 0, 255), 3)  # Alert

                                    # TODO: Capture and save image of the unattended bag
                                    # Check if the image is already captured
                                    if not region['image_captured']:
                                        # Capture and save image of the unattended bag
                                        bag_image = frame[int(y - h / 2):int(y + h / 2), int(
                                            x - w / 2):int(x + w / 2)].copy()
                                        save_image(bag_image, "unattended_bag_images",
                                                   f'unattended_bag_{time.time()}.jpg')
                                        # Set the image captured flag to True
                                        region['image_captured'] = True
                                    else:
                                        print("Image already captured")
                            else:
                                # Reset the unattended_bag_time if the distance is less than 100 pixels
                                region['unattended_bag_time'] = None

                                # Reset the image captured flag to False
                                region['image_captured'] = False

        # TODO: Draw regions (Polygons/Rectangles)
        for region in counting_regions:
            region_color = region['region_color']

            polygon_coords = np.array(region['polygon'].exterior.coords, dtype=np.int32)

            # TODO: To display the region area for counting
            cv2.polylines(frame, [polygon_coords], isClosed=True, color=region_color, thickness=2)


        yield frame
    cv2.destroyAllWindows()
