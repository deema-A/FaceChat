async_mode = None

if async_mode is None:
    try:
        import eventlet
        async_mode = 'eventlet'
    except ImportError:
        pass

    if async_mode is None:
        try:
            from gevent import monkey
            async_mode = 'gevent'
        except ImportError:
            pass

    if async_mode is None:
        async_mode = 'threading'

    print('async_mode is ' + async_mode)

# monkey patching is necessary because this application uses a background
# thread
if async_mode == 'eventlet':
    import eventlet
    eventlet.monkey_patch()
elif async_mode == 'gevent':
    from gevent import monkey
    monkey.patch_all()

from typing import Dict, List
import os
import torch
import time
from threading import Thread
import collections
import queue
from threading import Timer, Lock
from flask import Flask, Response, render_template, request
from flask_socketio import SocketIO, emit
import numpy as np

# audio processing
# from transformers import AutoProcessor, WhisperForConditionalGeneration
import scipy.signal as sps
from scipy.io import wavfile
import pyaudio
import webrtcvad
from vad import VAD, RepeatTimer
from langdetect import detect
from transformers import AutoProcessor, WhisperForConditionalGeneration
from TTS.api import TTS

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

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

def Int2Float(sound):
    _sound = np.copy(sound)  #
    abs_max = np.abs(_sound).max()
    _sound = _sound.astype("float32")
    if abs_max > 0:
        _sound *= 1 / abs_max
    audio_float32 = torch.from_numpy(_sound.squeeze())
    return audio_float32


class ASR:
    def __init__(self) -> None:
        model_name = "openai/whisper-medium"
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model_sample_rate = 16000
        self.device = torch.device("cpu")

    def __call__(self, data, sample_rate=16000) -> str:
        """
        Args:
            data: PCM float32 format
            sample_rate: the sample rate of data
        """
        is_valid = True
        # first, resample the data to the model's sample_rate
        if sample_rate != self.model_sample_rate:
            number_of_samples = round(len(data) * float(self.model_sample_rate) / sample_rate)
            data = sps.resample(data, number_of_samples)

        # genearte text
        inputs = self.processor(data, return_tensors="pt", sampling_rate=self.model_sample_rate)
        input_features = inputs.input_features.to(self.device)
        generated_ids = self.model.generate(inputs=input_features)
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        if len(generated_ids[0]) < 4:
            is_valid = False
        
        if not isEnglish(text):
            is_valid = False

        return text, is_valid

    def to(self, device):
        self.model = self.model.to(device)
        self.device = device
        return self

class TTSModel:
    def __init__(self) -> None:
        self.model = TTS("tts_models/en/vctk/vits", gpu=True)

    def __call__(self, text) -> np.float32:
        wav = self.model.tts(text, speaker=self.model.speakers[0])
        return wav


class AudioContext:
    """Streams raw audio from web microphone. Data is received in a separate thread, and stored in a buffer, to be read from.
    """

    MIC_RATE = 16000
    CHANNELS = 1
    BLOCKS_PER_SECOND = 50

    def __init__(self):
        self.buffer_queue = queue.Queue()

    def read(self):
        """Return a block of audio data, blocking if necessary."""
        return self.buffer_queue.get()

    def update(self, audioData):
        """Update the audio buffer."""
        self.buffer_queue.put(audioData)
        


asr_model = ASR()
tts_model = TTSModel()
# specify the running device
device = torch.device("cuda")
asr_model = asr_model.to(device)

buffer_queue = queue.Queue()

@socketio.on("audio_listen")
def audio_listen(audioData):
    global audio_context
    global buffer_queue
    # audio_context.update(audioData)
    buffer_queue.put(audioData)


@socketio.on("system_init")
def system_init(audioData):
    # speak
    text = "Greetings!"
    
    audio_float32 = tts_model(text)
    audio_int16 = (np.array(audio_float32, dtype=np.float32) * 32768).astype(np.int16)

    wav_header = genWaveHeader(sampleRate=22050, bitsPerSample=16, channels=1, samples=len(audio_int16))

    # return Response(sound(), mimetype="audio/x-wav")
    # data = np.array(data, dtype=np.float32).tobytes()
    speak_data = wav_header + audio_int16.tobytes()
    
    socketio.emit('audio_speak', speak_data)
    print("sending data!")

    time.sleep((len(audio_int16) + 100) / 22050)

    text = "I am an Talking AI Agent! I can recognize your speech and speak back to you!"
    audio_float32 = tts_model(text)
    audio_int16 = (np.array(audio_float32, dtype=np.float32) * 32768).astype(np.int16)

    wav_header = genWaveHeader(sampleRate=22050, bitsPerSample=16, channels=1, samples=len(audio_int16))

    # return Response(sound(), mimetype="audio/x-wav")
    # data = np.array(data, dtype=np.float32).tobytes()
    speak_data = wav_header + audio_int16.tobytes()
    
    socketio.emit('audio_speak', speak_data)
    print("sending data!")


@app.route("/", methods=["POST", "GET"])
def index():
    print("intialized")
    return render_template("index.html")


class VADAudio:
    """Filter & segment audio with voice activity detection."""

    def __init__(self, input_rate, audio_context):
        self.input_rate = input_rate
        self.audio_context = audio_context
        self.RATE_PROCESS = 16000
        self.block_size = 743
        self.frame_duration_ms = 1000 * self.block_size // self.input_rate
        self.sample_rate = 16000
        self.silence_duration_ms = 500

        self.vad = webrtcvad.Vad(mode=3)

    # def frame_generator(self):
    #     """Generator that yields all audio frames from microphone."""
    #     while True:
    #         yield self.audio_context.read()

    def vad_collector(self, padding_ms=300, ratio=0.75, frames=None):
        """Generator that yields series of consecutive audio frames comprising each utterence, separated by yielding a single None.
            Determines voice activity by ratio of frames in padding_ms. Uses a buffer to include padding_ms prior to being triggered.
            Example: (frame, ..., frame, None, frame, ..., frame, None, ...)
                      |---utterence---|        |---utterence---|
        """
        global vad_model
        global vad_iterator
        global buffer_queue
        # if frames is None:
        #     frames = self.frame_generator()

        num_padding_frames = padding_ms // self.frame_duration_ms
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False
        empty_frame_count = 0
        max_empty_frame_count = self.silence_duration_ms // self.frame_duration_ms

        while True:
            frame = buffer_queue.get()

            # # is_speech = self.vad.is_speech(frame, self.sample_rate)
            # chunk = np.frombuffer(frame, np.int16)
            # # audio_float32 = torch.from_numpy(audio_float32)
            # audio_float32 = Int2Float(chunk)

            # speech_prob = vad_model(audio_float32, self.sample_rate).item()
            # is_speech = speech_prob > 0.5

            is_speech = self.vad.is_speech(frame[-960:], self.sample_rate)

            if not triggered:
                ring_buffer.append((frame, is_speech))
                # if speaking
                num_voiced = len([f for f, is_speech in ring_buffer if is_speech])
                if num_voiced > ratio * ring_buffer.maxlen:
                    triggered = True
                    for frame, is_speech in ring_buffer:
                        yield frame
                    ring_buffer.clear()
            else:
                yield frame
                ring_buffer.append((frame, is_speech))
                # if not seapking
                num_unvoiced = len([f for f, is_speech in ring_buffer if not is_speech])
                if num_unvoiced > ratio * ring_buffer.maxlen:
                    # detects 5 consecutive empty frames
                    if empty_frame_count > max_empty_frame_count:
                        triggered = False
                        yield None
                        ring_buffer.clear()
                        empty_frame_count = 0
                    else:
                        empty_frame_count += 1
                else:
                    # reset empty_frame_count if detects speech
                    empty_frame_count = 0


class EngagementDetector(Thread):
    def __init__(self, audio_context):
        Thread.__init__(self)
        self.audio_context = audio_context
        self.vad_audio = VADAudio(input_rate=16000, audio_context=self.audio_context)
        self.vad_model, vad_utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad")
        (self.get_speech_ts, save_audio, read_audio, VADIterator, collect_chunks) = vad_utils
        self.count = 0
        

    def run(self):
        frames = self.vad_audio.vad_collector()
        wav_data = bytearray()

        vad_model, vad_utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad")
        (get_speech_ts, save_audio, read_audio, VADIterator, collect_chunks) = vad_utils


        print("Listening...")

        for frame in frames:
            # print("reading frame")
            if frame is not None:
                wav_data.extend(frame)                
            else:
                data = np.frombuffer(wav_data, np.int16)
                data = Int2Float(data)

                # two-stage VAD
                time_stamps = get_speech_ts(data, vad_model)

                if len(time_stamps) > 0:
                    print("Speaking:", end="")
                    text, is_asr_valid = asr_model(data, sample_rate=16000)
                    print(text)

                    if is_asr_valid:
                        # speak
                        audio_float32 = tts_model(text)
                        audio_int16 = (np.array(audio_float32, dtype=np.float32) * 32768).astype(np.int16)

                        wav_header = genWaveHeader(sampleRate=22050, bitsPerSample=16, channels=1, samples=len(audio_int16))

                        # return Response(sound(), mimetype="audio/x-wav")
                        # data = np.array(data, dtype=np.float32).tobytes()
                        speak_data = wav_header + audio_int16.tobytes()
                        
                        socketio.emit('audio_speak', speak_data)
                        print("sending data!")

                        # socketio.emit("audio_response", {"data": text})

                        # clear buffer
                        wav_data = bytearray()


audio_context = AudioContext()
engagement_detector = EngagementDetector(audio_context)
engagement_detector.start()


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=51000, debug=False, 
    keyfile="key.pem", certfile="cert.pem"
    )
    # app.run(host='0.0.0.0', debug=True, threaded=True, port=9900, ssl_context=("cert.pem", "key.pem"))
