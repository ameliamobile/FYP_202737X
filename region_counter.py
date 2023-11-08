import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import lap
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.point import Point

from ultralytics import YOLO
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors
#
# track_history = defaultdict(list)
#
# line_thickness = 2
# track_thickness = 2
# region_thickness = 2
# view_img = False
#
# current_region = None
# counting_regions = [
#     {
#         'name': 'YOLOv8 Rectangle Region',
#         'polygon': Polygon([(200, 250), (440, 250), (440, 550), (200, 550)]),  # Polygon points
#         'counts': 0,
#         'dragging': False,
#         'region_color': (37, 255, 225),  # BGR Value
#         'text_color': (0, 0, 0),  # Region Text Color
#     }, ]
#
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
#
#     """
#         Run Region counting on a video using YOLOv8 and ByteTrack.
#
#         Supports movable region for real time counting inside specific area.
#         Supports multiple regions counting.
#         Regions can be Polygons or rectangle in shape
#
#         """
#
#
# def region_detection(path_x):
#     vid_frame_count = 0
#
#     # Video setup
#     video_capture = path_x
#     cap = cv2.VideoCapture(video_capture)
#     frame_width = int(cap.get(3))
#     frame_height = int(cap.get(4))
#
#     # Setup Model
#     model = YOLO(" ../YOLO-Weights/yolov8n.pt")
#
#     # Iterate over video frames
#     while cap.isOpened():
#     #while True:
#         success, frame = cap.read()
#         if not success:
#             break
#         vid_frame_count += 1
#
#         # Extract the results
#         results = model.track(frame, persist=True)
#
#         if results[0].boxes.id is not None:
#             boxes = results[0].boxes.xywh.cpu()
#             track_ids = results[0].boxes.id.int().cpu().tolist()
#             clss = results[0].boxes.cls.cpu().tolist()
#             names = results[0].names
#
#             annotator = Annotator(frame, line_width=line_thickness, example=str(names))
#
#             for box, track_id, cls in zip(boxes, track_ids, clss):
#                 x, y, w, h = box
#                 label = str(names[cls])
#                 xyxy = (x - w / 2), (y - h / 2), (x + w / 2), (y + h / 2)
#
#                 # Bounding box plot
#                 bbox_color = colors(cls, True)
#                 annotator.box_label(xyxy, label, color=bbox_color)
#
#                 # Tracking Lines plot
#                 track = track_history[track_id]
#                 track.append((float(x), float(y)))
#                 if len(track) > 30:
#                     track.pop(0)
#                 points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
#                 cv2.polylines(frame, [points], isClosed=False, color=bbox_color, thickness=track_thickness)
#
#                 # Check if detection inside region
#                 for region in counting_regions:
#                     if region['polygon'].contains(Point((x, y))):
#                         region['counts'] += 1
#
#         # Draw regions (Polygons/Rectangles)
#         for region in counting_regions:
#             region_label = str(region['counts'])
#             region_color = region['region_color']
#             region_text_color = region['text_color']
#
#             polygon_coords = np.array(region['polygon'].exterior.coords, dtype=np.int32)
#             centroid_x, centroid_y = int(region['polygon'].centroid.x), int(region['polygon'].centroid.y)
#
#             text_size, _ = cv2.getTextSize(region_label,
#                                            cv2.FONT_HERSHEY_SIMPLEX,
#                                            fontScale=0.7,
#                                            thickness=line_thickness)
#             text_x = centroid_x - text_size[0] // 2
#             text_y = centroid_y + text_size[1] // 2
#             cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5),
#                           region_color, -1)
#             cv2.putText(frame, region_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region_text_color,
#                         line_thickness)
#             cv2.polylines(frame, [polygon_coords], isClosed=True, color=region_color, thickness=region_thickness)
#
#         if view_img:
#             if vid_frame_count == 1:
#                 cv2.namedWindow('Ultralytics YOLOv8 Region Counter Movable')
#                 cv2.setMouseCallback('Ultralytics YOLOv8 Region Counter Movable', mouse_callback)
#             cv2.imshow('Ultralytics YOLOv8 Region Counter Movable', frame)
#
#
#         #for region in counting_regions:  # Reinitialize count for each region
#             #region['counts'] = 0
#
#
#         #del vid_frame_count
#         yield frame
#     cv2.destroyAllWindows()


