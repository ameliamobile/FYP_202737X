"""Development of a Web Application for the Deployment of a Deep Learning Project"""

from flask import Flask, render_template, Response, session

#FlaskForm--> it is required to receive input from the user
# Whether uploading a video file  to our object detection model

from flask_wtf import FlaskForm


from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired
import os

# Required to run the YOLOv8 model
import cv2

# YOLO_Video is the python file which contains the code for our object detection model
# Video Detection is the Function which performs Object Detection on Input Video
from YOLO_media import file_detection
from region_counter import unitv2_detection
from bag_detection import camera_detection

app = Flask(__name__)

app.config['SECRET_KEY'] = 'amelia'
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# TODO: Use FlaskForm to get input video file  from user
class UploadFileForm(FlaskForm):
    # Store the uploaded video file path in the FileField in the variable file
    # Add validators to make sure the user inputs the video in the valid format and user does upload the video when
    # prompted to do so
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Run")

# TODO: For media input
def generate_frames(path_x=''):
    yolo_output = file_detection(path_x)
    for detection_ in yolo_output:
        ref, buffer = cv2.imencode('.jpg', detection_)

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')

# TODO: For Unitv2 camera stream
def generate_frames_unit():
    yolo_output = unitv2_detection()
    for detection_ in yolo_output:
        ref, buffer = cv2.imencode('.jpg', detection_)

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')

# TODO: For PC camera stream
def generate_frames_count(path_x):
    yolo_output = camera_detection(path_x)
    for detection_ in yolo_output:
        ref, buffer = cv2.imencode('.jpg', detection_)

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')





@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    session.clear()
    return render_template('index.html')
# Rendering the Webcam Rage
# Webcam page for the application

# TODO: Video Page for the application
@app.route('/mediapage', methods=['GET','POST'])
def front():
    # Upload File Form: Create an instance for the Upload File Form
    form = UploadFileForm()
    if form.validate_on_submit():
        # Uploaded video file path is saved here
        file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                               secure_filename(file.filename)))  # Then save the file
        # Use session storage to save video file path
        session['video_path'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                                             secure_filename(file.filename))
    return render_template('media.html', form=form)

@app.route("/unitpage", methods=['GET','POST'])
def unitcam():
    session.clear()
    return render_template('unit.html')

@app.route("/webpage", methods=['GET','POST'])
def webcam():
    session.clear()
    return render_template('web.html')




# TODO: For media input
@app.route('/mediaapp')
def media():
    return Response(generate_frames(path_x=session.get('video_path', None)), mimetype='multipart/x-mixed-replace; boundary=frame')

# TODO: To display the Live Feed of the UnitV2 feed on Camera page
@app.route('/unitv2app')
def unitv2():
    return Response(generate_frames_unit(), mimetype='multipart/x-mixed-replace; boundary=frame')


# TODO: To display the Live Feed on Camera page (region counter)
@app.route('/webcameraapp')
def webcamera():
    return Response(generate_frames_count(path_x=0), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)

