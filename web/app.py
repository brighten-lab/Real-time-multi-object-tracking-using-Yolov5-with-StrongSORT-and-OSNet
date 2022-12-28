from flask import Flask, render_template, request, Response
import pymysql
import cv2

app = Flask(__name__)

cam1 = cv2.VideoCapture('rtsp://brighten:brighten0701@192.168.0.22/stream1')
cam2 = cv2.VideoCapture('rtsp://brighten:brighten0701@192.168.0.44/stream_ch00_0')
cam3 = cv2.VideoCapture('rtsp://')
cam4 = cv2.VideoCapture('rtsp://')

@app.route('/')
def index():
    return render_template('video.html')
    
@app.route('/video_feed1')
def video_feed1():
    return Response(gen_frames1(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed2():
    return Response(gen_frames2(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed3')
def video_feed3():
    return Response(gen_frames3(), mimetype='multipart/x-mixed-replace; boundary=frame')    

@app.route('/video_feed4')
def video_feed4():
    return Response(gen_frames4(), mimetype='multipart/x-mixed-replace; boundary=frame')


def gen_frames1():
    while True:
        success, frame = cam1.read()  # read the camera frame
        if not success:
            yield(None)
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def gen_frames2():
    while True:
        success, frame = cam2.read()  # read the camera frame
        if not success:
            yield(None)
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def gen_frames3():
    while True:
        success, frame = cam3.read()  # read the camera frame
        if not success:
            yield(None)
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def gen_frames4():
    while True:
        success, frame = cam4.read()  # read the camera frame
        if not success:
            yield(None)
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)