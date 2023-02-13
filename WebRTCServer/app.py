from flask import Flask, render_template, request, Response
from flask_socketio import SocketIO, emit
import time
import io
from PIL import Image
import base64
import numpy as np
from flask_cors import CORS, cross_origin
from engineio.payload import Payload
from werkzeug.utils import secure_filename
import pickle

import soundfile

# import imutils
# import dlib
# import pyshine as ps

# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
Payload.max_decode_packets = 2048

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')


import pyaudio

FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5

 
audio1 = pyaudio.PyAudio()


def genWaveHeader(sampleRate, bitsPerSample, channels, samples):
    datasize = samples * channels * bitsPerSample // 8
    o = bytes("RIFF",'ascii')                                               # (4byte) Marks file as RIFF
    o += (datasize + 36).to_bytes(4,'little')                               # (4byte) File size in bytes excluding this and RIFF marker
    o += bytes("WAVE",'ascii')                                              # (4byte) File type
    o += bytes("fmt ",'ascii')                                              # (4byte) Format Chunk Marker
    o += (16).to_bytes(4,'little')                                          # (4byte) Length of above format data
    o += (1).to_bytes(2,'little')                                           # (2byte) Format type (1 - PCM)
    o += (channels).to_bytes(2,'little')                                    # (2byte)
    o += (sampleRate).to_bytes(4,'little')                                  # (4byte)
    o += (sampleRate * channels * bitsPerSample // 8).to_bytes(4,'little')  # (4byte)
    o += (channels * bitsPerSample // 8).to_bytes(2,'little')               # (2byte)
    o += (bitsPerSample).to_bytes(2,'little')                               # (2byte)
    o += bytes("data",'ascii')                                              # (4byte) Data Chunk Marker
    o += (datasize).to_bytes(4,'little')                                    # (4byte) Data size in bytes
    return o
    

@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index.html')

# @app.route("/audio_feed")
# def audio_feed():
#     """Audio streaming route. Put this in the src attribute of an audio tag."""
#     return Response(generateAudio(),
#                     mimetype="audio/x-wav")


# @app.route('/post_stream', methods=['POST'])
# def post_stream():
#     """
#     Read in streaming wav data.
#     VAD detection should be put here
#     """
#     global count
#     blobData = request.files['data']
#     print("data coming:", blobData)
#     data, rate = soundfile.read(io.BytesIO(blobData.read()))
#     blobData.close()
#     return render_template('index.html')


# count = 0


# def generateAudio():
#     """Audio streaming generator function."""
#     currChunk = audioRec.record()
#     data_to_stream = genHeader(44100, 32, 1, 200000) + currChunk
#     yield data_to_stream

# @socketio.on('audio_back')
# def audio_stream_back(data):
#     # start sending
#     def sound():
#         data = wav_header
#         data += stream.read(CHUNK)
#         yield(data)
#         while True:
#             data = stream.read(CHUNK)
#             yield(data)
#     return Response(sound(), mimetype="audio/x-wav")

# @socketio.on('stream')
# def process_stream(data):
#     global count
#     print("data received!")
#     # # read audio
#     # idx = data.find('base64,')
#     # base64_string = base64_string[idx + 7:]
#     # sbuf = io.BytesIO()
#     # sbuf.write(base64.b64decode(base64_string, ' /'))

#     with open(f"temp/{count}.base64", "w") as f:
#         # bytes = base64.b64decode(data)
#         f.write(data)
#         count += 1

#     return render_template('index.html')

@app.route('/audio')
def audio():
    # start Recording
    def sound():

        CHUNK = 1024
        sampleRate = 44100
        bitsPerSample = 16
        channels = 2
        wav_header = genWaveHeader(sampleRate, bitsPerSample, channels)

        stream = audio1.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,input_device_index=1,
                        frames_per_buffer=CHUNK)
        print("recording...")
        #frames = []
        first_run = True
        while True:
           if first_run:
               data = wav_header + stream.read(CHUNK)
               first_run = False
           else:
               data = stream.read(CHUNK)
           yield(data)

    return Response(sound())



if __name__ == '__main__':
    socketio.run(app,
                 host='0.0.0.0',
                 port=9000,
                 debug=True,
                 keyfile='key.pem',
                 certfile='cert.pem')

    # app.run(host='0.0.0.0', debug=True, threaded=True, port=9900, ssl_context=("cert.pem", "key.pem"))