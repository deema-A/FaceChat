from flask import Flask, Response, render_template, request
from flask_socketio import SocketIO, emit
import time
import io
from PIL import Image
import base64, cv2
import numpy as np
from flask_cors import CORS, cross_origin
from engineio.payload import Payload
from werkzeug.utils import secure_filename
import pickle
# audio processing
import pyaudio
import wave
import soundfile
from scipy.io import wavfile





from audio_format import byte_to_float


# import imutils
# import dlib
# import pyshine as ps

# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
Payload.max_decode_packets = 2048

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')


def genWaveHeader(sampleRate, bitsPerSample, channels, samples):
    datasize = samples * channels * bitsPerSample // 8
    o = bytes("RIFF", 'ascii')  # (4byte) Marks file as RIFF
    o += (datasize + 36).to_bytes(
        4,
        'little')  # (4byte) File size in bytes excluding this and RIFF marker
    o += bytes("WAVE", 'ascii')  # (4byte) File type
    o += bytes("fmt ", 'ascii')  # (4byte) Format Chunk Marker
    o += (16).to_bytes(4, 'little')  # (4byte) Length of above format data
    o += (1).to_bytes(2, 'little')  # (2byte) Format type (1 - PCM)
    o += (channels).to_bytes(2, 'little')  # (2byte)
    o += (sampleRate).to_bytes(4, 'little')  # (4byte)
    o += (sampleRate * channels * bitsPerSample // 8).to_bytes(
        4, 'little')  # (4byte)
    o += (channels * bitsPerSample // 8).to_bytes(2, 'little')  # (2byte)
    o += (bitsPerSample).to_bytes(2, 'little')  # (2byte)
    o += bytes("data", 'ascii')  # (4byte) Data Chunk Marker
    o += (datasize).to_bytes(4, 'little')  # (4byte) Data size in bytes
    return o

audio1 = pyaudio.PyAudio()


FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 128
RECORD_SECONDS = 5
 
bitsPerSample = 16

wav_header = genWaveHeader(RATE, bitsPerSample, CHANNELS, 2000*10**6)

stream = audio1.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)

@app.route('/audio_back')
def audio_back():
    # start Recording
    def sound():
        print("recording...")
        #frames = []
        first_run = True
        while True:
           if first_run:
               data = wav_header + stream.read(CHUNK)
               first_run = True
           else:
               data = stream.read(CHUNK)
           yield(data)
           print("sending data!")

    return Response(sound())

# stream = audio1.open(format=FORMAT, channels=CHANNELS,
#                 rate=RATE, input=True,
#                 frames_per_buffer=CHUNK)

# @socketio.on('audio_back')
# def audio_back(data):
#     # breakpoint()
#     audio_data = stream.read(CHUNK)
#     # audio_data = byte_to_float(audio_data)
#     emit("audio_back", audio_data)

@app.route('/', methods=['POST', 'GET'])
def index():
    print("intialized")
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app,
                 host='0.0.0.0',
                 port=9000,
                 debug=True,
                 keyfile='key.pem',
                 certfile='cert.pem')
    # app.run(host='0.0.0.0', debug=True, threaded=True, port=9900, ssl_context=("cert.pem", "key.pem"))