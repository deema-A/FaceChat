import os
import torch
import time
from threading import Timer
# every 1 second check if the user is speaking


class RepeatTimer(Timer):
    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)



SAMPLE_RATE = 16000
VAD_WINDOW_SIZE = 1536 * 3
VAD_THRESHOLD = 0.5
# ms
VAD_RUN_EVERY_SECOND = 0.5
VAD_HISTORY_SECOND = 2
USE_ONNX = True  # change this to True if you want to test onnx model

BETA = 0.5

class VAD:
    def __init__(self, audio_buffer, sampleRate) -> None:
        """Initialize the VAD class
            Args:
                audio_buffer: a list of audio data shared between the main thread and the VAD thread
        """
        self.audio_buffer = audio_buffer
        self.sampleRate = sampleRate
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                      model='silero_vad',
                                      force_reload=True,
                                      onnx=USE_ONNX)

        (get_speech_timestamps, save_audio, read_audio, VADIterator,
         collect_chunks) = utils

        self.model = model
        self.vad_iterator = VADIterator(model)
        self.read_audio = read_audio

        # store the history activity scores
        self._buffer = []
        self._past_scores = []

    def run_vad(self):
        """Check if the user is speaking"""
        window_size_samples = VAD_WINDOW_SIZE
        # process buffer
        # use the last 2 seconds of audio data
        self._buffer = self.audio_buffer[-self.sampleRate*VAD_HISTORY_SECOND:]

        speech_probs = []
        prev_speech_prob = 0.0
        
        # for i in range(0, len(self._buffer), window_size_samples):
        #     chunk = self._buffer[i: i + window_size_samples]
        #     if len(chunk) < window_size_samples:
        #         break
        #     chunk = torch.FloatTensor(chunk)
        #     speech_prob = self.model(chunk, self.sampleRate).item()
            
        #     speech_probs.append(prev_speech_prob * BETA + speech_prob * (1 - BETA))
        #     prev_speech_prob = speech_prob
        
        # # smooth the speech probability
        # if self._past_scores and speech_probs:
        #     score = speech_probs[-1]
        # else:
        #     score = prev_speech_prob
        # self._past_scores.append(score)

        speech_prob = self.model(chunk, self.sampleRate).item()
        self._past_scores.append(speech_prob)

        # self.vad_iterator.reset_states()
        self._buffer = []
        print(score)

    def is_speech(self) -> bool:
        """Check if the audio data is speech"""
        if self._past_scores:
            return self._past_scores[-1]
        else:
            return 0.0


if __name__ == "__main__":
    audio_buffer = []
    vad = VAD(audio_buffer)
    timer = RepeatTimer(VAD_RUN_EVERY_SECOND, vad.run_vad)
    timer.start()

    wav = vad.read_audio(f'en_example.wav', sampling_rate=SAMPLE_RATE)

    window_second = 100 
    window_size = int(SAMPLE_RATE * window_second / 1000)

    for i in range(0, len(wav), window_size):
        chunk = wav[i: i+ window_size].tolist()
        if len(chunk) < window_size:
            break
        audio_buffer.extend(chunk)
        time.sleep(window_second/1000)
        #print(vad.is_speech())

    timer.cancel()
    timer.join()
    # while True:
    #     # get audio data
    #     # vad.update_audio(data)
    #     time.sleep(0.1)