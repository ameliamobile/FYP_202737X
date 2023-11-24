"""Region Counter and Unattended Bag Alert Code"""
# TODO: To implement Object detection and Region counter in a specified region of interest on a video stream
# TODO: Baggage Tracking:
#  Implement object tracking to monitor the movement of unattended baggage.
#  Using the centroid of the detected object to calculate the distance between person and chair
#  To determine whether the suitcase is left unattended
#  Trigger alerts if baggage remains unattended for an extended period.

import cv2
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
        'polygon': Polygon([(320, 0), (640, 0), (640, 480), (320, 480)]),  #The whole frame
        'counts': 0,
        'dragging': False,
        'region_color': (37, 255, 225),  # BGR Value
        'text_color': (0, 0, 0),  # Region Text Color
    },
    ]

def unitv2_detection():
    vid_frame_count = 0

    # TODO: Video Setup
    cap = cv2.VideoCapture('http://10.254.239.1/video_feed')
    cap.set(cv2.CAP_PROP_FPS, 30)  # Desired frame rate
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))

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
                x, y, w, h = box
                label = str(names[cls])
                xyxy = (x - w / 2), (y - h / 2), (x + w / 2), (y + h / 2)

                # Bounding box plot
                bbox_color = colors(cls, True)
                annotator.box_label(xyxy, label, color=bbox_color)

                # Tracking Lines plot
                track = track_history[track_id]
                track.append((float(x), float(y)))
                if len(track) > 30:
                    track.pop(0)
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=bbox_color, thickness=2)

                # Check if detection inside region
                for region in counting_regions:
                    if region['polygon'].contains(Point((x, y))):
                        region['counts'] += 1

            # Draw regions (Polygons/Rectangles)
        for region in counting_regions:
            region_label = str(region['counts'])
            region_color = region['region_color']
            region_text_color = region['text_color']

            polygon_coords = np.array(region['polygon'].exterior.coords, dtype=np.int32)
            centroid_x, centroid_y = int(region['polygon'].centroid.x), int(region['polygon'].centroid.y)

            text_size, _ = cv2.getTextSize(region_label,
                                           cv2.FONT_HERSHEY_SIMPLEX,
                                           fontScale=0.7,
                                           thickness=2)
            text_x = centroid_x - text_size[0] // 2
            text_y = centroid_y + text_size[1] // 2
            cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5),
                          region_color, -1)
            cv2.putText(frame, region_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region_text_color,
                        2)
            cv2.polylines(frame, [polygon_coords], isClosed=True, color=region_color, thickness=2)

        # TODO: Reinitialize count for each region
        for region in counting_regions:
            region['counts'] = 0

        yield frame
    cv2.destroyAllWindows()

