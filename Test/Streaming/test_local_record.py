import os
from queue import Queue
import sys
import json
import threading

import pyaudio
import wave

FORMAT = pyaudio.paInt16
CHANNELS = 1
CHUNK = 1024
RATE = 44100


class StreamPlayer(threading.Thread):
    def __init__(self, queue):
        """ Init audio stream """
        threading.Thread.__init__(self) 

        self.queue = queue
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            output=True,
        )

    def run(self):
        silence = chr(0)*CHUNK*CHANNELS*2 
        while True:
            # data = self.queue.get_nowait()
            try:
                data = self.queue.get(False)
            except:
                data = silence
                # if data == '' or data is None:
                #     data = silence
            self.stream.write(data)

queue = Queue()
audio = pyaudio.PyAudio()

player = StreamPlayer(queue)
player.setDaemon(True)
player.start()

record_stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK,
)

try:
    frames = []
    print("* echoing")
    print("Press CTRL+C to stop")
    while True:
        data = record_stream.read(CHUNK, exception_on_overflow=False)
        queue.put(data)
    print("* done echoing")
except KeyboardInterrupt:
    record_stream.stop_stream()
    record_stream.close()
    audio.terminate()
    player.join()


# p = pyaudio.PyAudio()  # Create an interface to PortAudio

# print("Recording")


# p = pyaudio.PyAudio()
# info = p.get_host_api_info_by_index(0)
# numdevices = info.get('deviceCount')

# for i in range(0, numdevices):
#     if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
#         print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))

# stream = p.open(format=sample_format, channels=channels, rate=fs, frames_per_buffer=chunk, input=True, input_device_index=19)
# frames = []  # Initialize array to store frames

# # Store data in chunks for 3 seconds
# for i in range(0, int(fs / chunk * seconds)):
#     data = stream.read(chunk)
#     frames.append(data)

# # Stop and close the stream
# stream.stop_stream()
# stream.close()
# # Terminate the PortAudio interface
# p.terminate()

# print("Finished recording")

# # Save the recorded data as a WAV file
# wf = wave.open(filename, "wb")
# wf.setnchannels(channels)
# wf.setsampwidth(p.get_sample_size(sample_format))
# wf.setframerate(fs)
# wf.writeframes(b"".join(frames))
# wf.close()


# class AudioFile:
#     chunk = 1024

#     def __init__(self, file):
#         """ Init audio stream """
#         # super().__init__(self)

#         self.wf = wave.open(file, "rb")
#         self.p = pyaudio.PyAudio()
#         self.stream = self.p.open(
#             format=self.p.get_format_from_width(self.wf.getsampwidth()),
#             channels=self.wf.getnchannels(),
#             rate=self.wf.getframerate(),
#             output=True,
#         )

#     # def run(self):
#     #     pass

#     def play(self):
#         """ Play entire file """
#         data = self.wf.readframes(self.chunk)
#         while data != b"":
#             self.stream.write(data)
#             data = self.wf.readframes(self.chunk)

#     def close(self):
#         """ Graceful shutdown """
#         self.stream.close()
#         self.p.terminate()



# # Usage example for pyaudio
# a = AudioFile(filename)
# a.play()
# a.close()
