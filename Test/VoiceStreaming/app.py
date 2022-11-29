from flask import Flask, Response,render_template
import pyaudio

app = Flask(__name__)


FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5

 
audio1 = pyaudio.PyAudio()
 


@app.route("/play_wav")
def streamwav():
    def generate():
        with open("test.wav", "rb") as fwav:
            data = fwav.read(1024)
            while data:
                yield data
                data = fwav.read(1024)
    return Response(generate(), mimetype="audio/x-wav")

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

      
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False, threaded=True,port=5000)