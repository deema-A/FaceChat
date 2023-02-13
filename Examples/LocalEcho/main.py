import pyaudio
import json

FORMAT = pyaudio.paInt16
CHANNELS = 1
CHUNK = 128
RATE = 44100

audio = pyaudio.PyAudio()

for i in range(audio.get_device_count()):
    print(json.dumps(audio.get_device_info_by_index(i), indent=2))

stream = audio.open(format              = FORMAT,
                    channels            = CHANNELS,
                    rate                = RATE,
                    input               = True,
                    output              = True,
                    # input_device_index  = 27,
                    # output_device_index = 1,
                    frames_per_buffer   = CHUNK)

try:
    frames = []
    print("* echoing")
    print("Press CTRL+C to stop")
    
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

        if len(frames) > 0:
            stream.write(frames.pop(0), CHUNK)

    print("* done echoing")

except KeyboardInterrupt:
    stream.stop_stream()
    stream.close()
    audio.terminate()