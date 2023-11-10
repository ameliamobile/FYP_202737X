# Trying to integrate my code and region counter

from collections import defaultdict
#from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.point import Point

from ultralytics import YOLO
#from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors

track_history = defaultdict(list)

current_region = None
counting_regions = [
    {
        'name': 'YOLOv8 Rectangle Region',
        'polygon': Polygon([(100, 100), (540, 100), (540, 400), (100, 400)]),  # Polygon points
        'person_counts': 0,
        'chair_counts': 0,
        'dragging': False,
        'region_color': (37, 255, 225),  # BGR Value
        'text_color': (0, 0, 0),  # Region Text Color
    },
    ]

def region_detection(path_x):
    vid_frame_count = 0

    # Video setup
    video_capture = path_x
    cap = cv2.VideoCapture(video_capture)
    # cap = cv2.VideoCapture(
    #     'https://dm0qx8t0i9gc9.cloudfront.net/watermarks/video/piShJKb/videoblocks-tel-aviv-israel-january-2018-passengers-walking-through-airport-terminal_rxbjdf5pm__2c365d4ce0ca2c27df6c887d19cd79f7__P360.mp4')
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Setup Model
    model = YOLO(" ../YOLO-Weights/yolov8n.pt")

    # Iterate over video frames
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        vid_frame_count += 1

        # Extract the results
        results = model.track(frame, persist=True)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()
            names = results[0].names

            annotator = Annotator(frame, line_width=2, example=str(names))

            for box, track_id, cls in zip(boxes, track_ids, clss):

                # Add this if statement to filter 'person' and 'suitcase'
                if names[cls] not in ['person', 'chair']:
                    continue

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
                        if label == 'person':
                            region['person_counts'] += 1
                        elif label == 'chair':
                            region['chair_counts'] += 1

        # Draw regions (Polygons/Rectangles)
        for region in counting_regions:
            region_label_person = str(region['person_counts'])
            region_label_chair = str(region['chair_counts'])
            region_color = region['region_color']
            region_text_color = region['text_color']

            polygon_coords = np.array(region['polygon'].exterior.coords, dtype=np.int32)
            centroid_x, centroid_y = int(region['polygon'].centroid.x), int(region['polygon'].centroid.y)

            # text_size, _ = cv2.getTextSize(region_label,
            #                                cv2.FONT_HERSHEY_SIMPLEX,
            #                                fontScale=0.7,
            #                                thickness=2)
            # text_x = centroid_x - text_size[0] // 2
            # text_y = centroid_y + text_size[1] // 2
            # cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5),
            #               region_color, -1)
            cv2.putText(frame, f'Person Count: {region_label_person}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        region_text_color, 2)
            cv2.putText(frame, f'Chair Count: {region_label_chair}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        region_text_color, 2)
            cv2.polylines(frame, [polygon_coords], isClosed=True, color=region_color, thickness=2)
            # cv2.putText(frame, f'Person Count: {person_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # cv2.putText(frame, f'Chair Count: {chair_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        for region in counting_regions:  # Reinitialize count for each region
            region['person_counts'] = 0
            region['chair_counts'] = 0



        yield frame
    cv2.destroyAllWindows()

