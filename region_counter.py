# Trying to integrate my code and region counter

from collections import defaultdict
from pathlib import Path

import cv2
import math
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.point import Point

from ultralytics import YOLO
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors

track_history = defaultdict(list)

current_region = None
counting_regions = [
    {
        'name': 'YOLOv8 Rectangle Region',
        'polygon': Polygon([(0, 0), (640, 0), (640, 480), (0, 480)]),  # The whole frame
        #'polygon': Polygon([(50, 50), (590, 50), (590, 430), (50, 430)]),  # 50 px offset of the frame
        'person_counts': 0,
        'chair_counts': 0,
        'dragging': False,
        'region_color': (37, 255, 225),  # BGR Value
        'text_color': (0, 0, 0),  # Region Text Color
    },
    ]

# def mouse_callback(event, x, y, flags, param):
#     """Mouse call back event."""
#     global current_region
#
#     # Mouse left button down event
#     if event == cv2.EVENT_LBUTTONDOWN:
#         for region in counting_regions:
#             if region['polygon'].contains(Point((x, y))):
#                 current_region = region
#                 current_region['dragging'] = True
#                 current_region['offset_x'] = x
#                 current_region['offset_y'] = y
#
#     # Mouse move event
#     elif event == cv2.EVENT_MOUSEMOVE:
#         if current_region is not None and current_region['dragging']:
#             dx = x - current_region['offset_x']
#             dy = y - current_region['offset_y']
#             current_region['polygon'] = Polygon([
#                 (p[0] + dx, p[1] + dy) for p in current_region['polygon'].exterior.coords])
#             current_region['offset_x'] = x
#             current_region['offset_y'] = y
#
#     # Mouse left button up event
#     elif event == cv2.EVENT_LBUTTONUP:
#         if current_region is not None and current_region['dragging']:
#             current_region['dragging'] = False


def region_detection(path_x):
    vid_frame_count = 0

    # Video setup
    video_capture = path_x
    cap = cv2.VideoCapture(video_capture)
    # cap = cv2.VideoCapture(
    #     'https://dm0qx8t0i9gc9.cloudfront.net/watermarks/video/piShJKb/videoblocks-tel-aviv-israel-january-2018-passengers-walking-through-airport-terminal_rxbjdf5pm__2c365d4ce0ca2c27df6c887d19cd79f7__P360.mp4')
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))

    # Setup Model
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
                # Add this if statement to filter 'person' and 'chair'
                if names[cls] not in ['person', 'chair']:
                    continue

                x, y, w, h = box
                class_names = str(names[cls])
                label = str("bag unattended")

                xyxy = (x - w / 2), (y - h / 2), (x + w / 2), (y + h / 2)

                # TODO: Bounding box plot
                bbox_color = colors(cls, True)
                annotator.box_label(xyxy, color=bbox_color)

                # TODO: Tracking Lines plot
                track = track_history[track_id]
                track.append((float(x), float(y)))
                if len(track) > 30:
                    track.pop(0)
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=bbox_color, thickness=2)

                # TODO: Check if detection inside region
                for region in counting_regions:
                    if region['polygon'].contains(Point((x, y))):
                        if class_names == 'person':
                            region['person_counts'] += 1
                            # Calculate centroid coordinates
                            centroid_x_person = (x + (x + w)) / 2
                            centroid_y_person = (y + (y + h)) / 2
                        elif class_names == 'chair':
                            region['chair_counts'] += 1
                            # Calculate centroid coordinates
                            centroid_x_chair = (x + (x + w)) / 2
                            centroid_y_chair = (y + (y + h)) / 2

                            # TODO: Calculate the distance between person and chair centroids
                            distance = math.sqrt(
                                (centroid_x_person - centroid_x_chair) ** 2 + (
                                            centroid_y_person - centroid_y_chair) ** 2)

                            # TODO: Check if the bag is unattended (centroid distance > 100 pixels)
                            if distance > 100:
                                annotator.box_label(xyxy, label, color=bbox_color)

        # TODO: Draw regions (Polygons/Rectangles)
        for region in counting_regions:
            region_label_person = str(region['person_counts'])
            region_label_chair = str(region['chair_counts'])
            region_color = region['region_color']
            region_text_color = region['text_color']

            polygon_coords = np.array(region['polygon'].exterior.coords, dtype=np.int32)

            cv2.putText(frame, f'Person Count: {region_label_person}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        region_text_color, 2)
            cv2.putText(frame, f'Chair Count: {region_label_chair}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        region_text_color, 2)
            # To display the region area for counting
            cv2.polylines(frame, [polygon_coords], isClosed=True, color=region_color, thickness=2)

            # cv2.setMouseCallback('Ultralytics YOLOv8 Region Counter Movable', mouse_callback)

        for region in counting_regions:  # Reinitialize count for each region
            region['person_counts'] = 0
            region['chair_counts'] = 0

        yield frame
    cv2.destroyAllWindows()

# TODO: Centroid detection of abandoned bag in the airport premises

# TODO: To detect the centroid of the person and chair within specified regions, you can calculate the centroid based
#  on the individual detected bounding boxes

# TODO: If distance between person and chair is above 100:
#  display the label = bag unattended only on the chair boundary



