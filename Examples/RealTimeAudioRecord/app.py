from typing import Dict, List
from flask import Flask, Response, render_template, request
from flask_socketio import SocketIO, emit
from PIL import Image
import numpy as np
# audio processing
import pyaudio
from scipy.io import wavfile

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5
CHANNELS = 1

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


@socketio.on('audio_request')
def audio_unlim(data):
    print("audio request")
    # start Recording
    rate, data = wavfile.read("test.wav")
    bitsPerSample = 16
    wav_header = genWaveHeader(rate,
                               bitsPerSample,
                               CHANNELS,
                               samples=len(data))

    # return Response(sound(), mimetype="audio/x-wav")
    # data = np.array(data, dtype=np.float32).tobytes()
    data = wav_header + data.tobytes()
    socketio.emit('audio_data', data)

    # with open("test.wav", "rb") as f:
    #     data = f.read()
    #     socketio.emit('audio_data', data)


buffer = []
count = 0


@socketio.on('audio_record')
def handle_audio(audioData: Dict[str, float]):
    # Process the audio data
    # receiving raw pcm data.
    global buffer
    global count
    buffer.extend(audioData.values())

    if len(buffer) > 48000 * 4:
        # save pcm data to file
        buffer = np.array(buffer, dtype=np.float32)
        buffer = (buffer * 32767).astype(np.int16)
        wav_header = genWaveHeader(sampleRate=48000,
                                   bitsPerSample=16,
                                   channels=1,
                                   samples=len(buffer))

        with open(f"test_{count}.wav", "wb") as f:
            f.write(wav_header + buffer.tobytes())

        buffer = []
        count += 1


@app.route('/', methods=['POST', 'GET'])
def index():
    print("intialized")
    return render_template('index.html')


if __name__ == '__main__':
    socketio.run(app,
                 host='0.0.0.0',
                 port=55555,
                 debug=True,
                 keyfile='key.pem',
                 certfile='cert.pem')
    # app.run(host='0.0.0.0', debug=True, threaded=True, port=9900, ssl_context=("cert.pem", "key.pem"))