from flask import Flask, render_template, Response, jsonify, request

# Required to run the YOLOv8 model
import cv2

# YOLO_Video is the python file which contains the code for object detection model
# Video Detection is the Function which performs the Object Detection on Input video
from YOLO_Video import video_detection

app = Flask(__name__)

app.config['SECRET_KEY'] = 'amelia'
# Generate_frames function takes path of input video file and gives us the output with the bounding boxes around detected objects

# Display the video with detection
def generate_frames(path_x= ''):
    # Yolo_output variable stores the output for each detected object
    # The output with bounding boxes around detected objects

    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref, buffer = cv2.imencode('.jpg', detection_)
        # Any flask application requires encoded image to be converted into bytes
        # We will display the individual frames using Yield keyword
        # We will loop over all individual frames and display them as video
        # When we want the individual frames to be replaced by subsequent frames the Content-Type, or Mini-Type will be used
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer + b'\r\n')


@app.route('/video')
def video():
    return Response(generate_frames(path_x='../uploads/speedcam.mp4'), mimetype='/x-mixed-replace; boundary=frame')

@app.route('/webcam')
def webcam():
    return Response(generate_frames(path_x=0), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)