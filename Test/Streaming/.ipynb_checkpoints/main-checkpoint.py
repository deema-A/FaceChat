import os
from flask import Flask, Response,render_template
import pyaudio

app = Flask(__name__)

@app.route('/')
def index():
    """Audio streaming home page."""
    return render_template('index.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, threaded=True,port=5000)