from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import time
import io
from PIL import Image
import base64, cv2
import numpy as np
from flask_cors import CORS, cross_origin
from engineio.payload import Payload

from deepface import DeepFace

# import imutils
# import dlib
# import pyshine as ps

# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
Payload.max_decode_packets = 2048

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')


@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index.html')


def readb64(base64_string):
    "decode base64"
    idx = base64_string.find('base64,')
    base64_string = base64_string[idx + 7:]
    sbuf = io.BytesIO()
    sbuf.write(base64.b64decode(base64_string, ' /'))
    pimg = Image.open(sbuf)
    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)


def moving_average(x):
    return np.mean(x)


@socketio.on('catch-frame')
def catch_frame(data):
    emit('response_back', data)


global fps, prev_recv_time, cnt, fps_array
fps = 30
prev_recv_time = 0
cnt = 0
fps_array = [0]

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

old_x, old_y, old_w, old_h = 0, 0, 0, 0
old_pred_emotion = ''
pred_skip_count = 0

@socketio.on('image')
def image(data_image):
    global fps, cnt, prev_recv_time, fps_array, pred_skip_count
    global old_x, old_y, old_w, old_h, old_pred_emotion
    recv_time = time.time()
    text = 'FPS: ' + str(fps)
    frame = (readb64(data_image))
    exec_start_time = time.time()
    

    # 12 ms
    obj = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

    emotions, scores = zip(*obj['emotion'].items())
    index = np.argmax(scores)
    pred_emotion = emotions[index]
    x, y, w, h = obj['region'].values()

    if pred_emotion in emotions and (x > 0 and y > 0):
        # draw face box
        pred_skip_count = 0
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame,
                    pred_emotion, (x - 10, y - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.9,
                    color=(255, 0, 0),
                    thickness=2)
        old_pred_emotion = pred_emotion
        old_x, old_y, old_w, old_h = x, y, w, h
    else:
        pred_skip_count += 1
        # print(pred_skip_count)

        if pred_skip_count < 50:
            cv2.rectangle(frame, (old_x, old_y), (old_x + old_w, old_y + old_h),
                        (255, 0, 0), 2)
            cv2.putText(frame,
                        old_pred_emotion, (old_x - 10, old_y - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.9,
                        color=(255, 0, 0),
                        thickness=2)
        else:
            old_x, old_y, old_w, old_h = 0, 0, 0, 0
            old_pred_emotion = ''
            pred_skip_count = 0

    cv2.putText(frame,
                'fps '+ str(fps), (10, 20),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(255, 0, 0),
                thickness=1)

    exec_end_time = time.time()

    cv2.putText(frame,
                f"latency {(exec_end_time - exec_start_time) * 1000:.2f} ms", (10, 40),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(255, 0, 0),
                thickness=1)

    # encode it into jpeg
    imgencode = cv2.imencode('.jpeg', frame, [cv2.IMWRITE_JPEG_QUALITY, 40])[1]

    # base64 encode
    stringData = base64.b64encode(imgencode).decode('utf-8')
    b64_src = 'data:image/jpeg;base64,'
    stringData = b64_src + stringData

    # emit the frame back
    emit('response_back', stringData)

    fps = 1 / (recv_time - prev_recv_time)
    fps_array.append(fps)
    fps = round(moving_average(np.array(fps_array)), 1)
    prev_recv_time = recv_time
    #print(fps_array)
    cnt += 1
    if cnt == 30:
        fps_array = [fps]
        cnt = 0


def getMaskOfLips(img, points):
    """ This function will input the lips points and the image
        It will return the mask of lips region containing white pixels
    """
    mask = np.zeros_like(img)
    mask = cv2.fillPoly(mask, [points], (255, 255, 255))
    return mask


# def changeLipstick(img, value):
#     """ This funciton will take img image and lipstick color RGB
#         Out the image with a changed lip color of the image
#     """

#     img = cv2.resize(img, (0, 0), None, 1, 1)
#     imgOriginal = img.copy()
#     imgColorLips = imgOriginal
#     imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = detector(imgGray)

#     for face in faces:
#         x1, y1 = face.left(), face.top()
#         x2, y2 = face.right(), face.bottom()

#         facial_landmarks = predictor(imgGray, face)
#         points = []
#         for i in range(68):
#             x = facial_landmarks.part(i).x
#             y = facial_landmarks.part(i).y
#             points.append([x, y])

#         points = np.array(points)
#         imgLips = getMaskOfLips(img, points[48:61])

#         imgColorLips = np.zeros_like(imgLips)

#         imgColorLips[:] = value[2], value[1], value[0]
#         imgColorLips = cv2.bitwise_and(imgLips, imgColorLips)

#         value = 1
#         value = value // 10
#         if value % 2 == 0:
#             value += 1
#         kernel_size = (6 + value, 6 + value)  # +1 is to avoid 0

#         weight = 1
#         weight = 0.4 + (weight) / 400
#         imgColorLips = cv2.GaussianBlur(imgColorLips, kernel_size, 10)
#         imgColorLips = cv2.addWeighted(imgOriginal, 1, imgColorLips, weight, 0)

#     return imgColorLips

if __name__ == '__main__':
    socketio.run(app,
                 host='0.0.0.0',
                 port=9000,
                 debug=True,
                 keyfile='key.pem',
                 certfile='cert.pem')
    # app.run(host='0.0.0.0', debug=True, threaded=True, port=9900, ssl_context=("cert.pem", "key.pem"))