// import MediaRecorder from 'opus-media-recorder';

const canvas = document.querySelector('.visualizer');
const canvasCtx = canvas.getContext("2d");


function _readAsync(blob, mode) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = e => resolve(reader.result);
    reader.onerror = reject;
    reader[mode](blob);
  });
}


function toBase64Async(blob) {
  return this._readAsync(blob, 'readAsDataURL').then(dataUri =>
    dataUri.replace(/data:[^;]+;base64,/, '')
  );
}


// flask socketio
var socket = io.connect(window.location.protocol + '//' + document.domain + ':' + location.port);
socket.on('connect', function () {
  console.log("Connected...!", socket.connected)
});

// global audio context
var audioCtx;
var mediaRecorder;
var audioChunks = [];


console.log("Script loaded!")

// Setup Video Streaming
if (navigator.mediaDevices.getUserMedia) {
  navigator.mediaDevices.getUserMedia({ audio: true })
    .then(stream => {
      audio_visualize(stream);
      send_server2(stream);
    })
    .catch(function (err0r) {
      console.log(err0r)
    });
} else {
  alert("WebRTC not supported");
}

function send_server2(stream) {
  const options = { mimeType: 'audio/wav' }
  const workerOptions = {
    OggOpusEncoderWasmPath: 'https://cdn.jsdelivr.net/npm/opus-media-recorder@latest/OggOpusEncoder.wasm',
    WebMOpusEncoderWasmPath: 'https://cdn.jsdelivr.net/npm/opus-media-recorder@latest/WebMOpusEncoder.wasm'
  };

  window.MediaRecorder = OpusMediaRecorder;
  recorder = new MediaRecorder(stream, options, workerOptions);
  recorder.start();
  console.log("recorder started");
  // event listener
  recorder.ondataavailable = (event) => {
    // chunks.push(event.data);
    chunks.push(event.data);

  };

  var count = 0;
  const FPS = 1;
  let chunks = []

  setInterval(() => {
    if (chunks.length > 0) {
      const blob = new Blob(chunks, { type: recorder.mimeType });
      console.log(blob)

      // Using form and post
      var formData = new FormData();
      formData.append("data", blob);

      $.ajax({
        type: 'POST',
        url: '/post_stream',
        data: formData,
        contentType: false,
        processData: false,
        success: function (result) {
          console.log('post data success');
        },
        error: function (err) {
          alert('sorry an error occured');
          console.log('error', err);
        }
      });

      recorder.stop();
      recorder.start();
      chunks = [];
      count = count + 1;
    } else {
      recorder.requestData();
    }
  }, 1000 / FPS);

}


function send_server(stream) {
  // Instantiate the media recorder.
  const options = {
    mimeType: "audio/webm"
  }
  let chunks = [];

  const mediaRecorder = new MediaRecorder(stream, options);
  mediaRecorder.start();
  console.log("recorder started");
  // event listener
  mediaRecorder.ondataavailable = (event) => {
    // chunks.push(event.data);
    chunks.push(event.data);
  };


  var count = 0;
  const FPS = 1;

  setInterval(() => {
    const blob = new Blob(chunks, { type: mediaRecorder.mimeType });
    console.log(blob)

    // Using form and post
    var formData = new FormData();
    formData.append("data", blob);

    $.ajax({
      type: 'POST',
      url: '/post_stream',
      data: formData,
      contentType: false,
      processData: false,
      success: function (result) {
        console.log('post data success');
      },
      error: function (err) {
        alert('sorry an error occured');
        console.log('error', err);
      }
    });

    // toBase64Async(blob).then(data => socket.emit('stream', data));
    // console.log("data sent!");
    // debugger;

    // const blob = new Blob(chunks, { type: "audio/wav; codecs=MS_PCM" });
    // var form = new FormData();
    // form.append('file', blob, FILENAME);
    
    // socket.emit('stream', blob);
    mediaRecorder.stop();
    mediaRecorder.start();
    chunks = [];
    count = count + 1;
  }, 1000 / FPS);
}


function send_data(stream) {

  // addEventListener("dataavailable", (event) => { });


  console.log("sending data");

  recordAudio = RecordRTC(stream, {
    type: 'audio',
    mimeType: 'audio/webm',
    sampleRate: 44100,
    desiredSampRate: 16000,

    recorderType: StereoAudioRecorder,
    numberOfAudioChannels: 1,


    //1)
    // get intervals based blobs
    // value in milliseconds
    // as you might not want to make detect calls every seconds
    timeSlice: 4000,


    //2)
    // as soon as the stream is available
    ondataavailable: function (blob) {

      // 3
      // making use of socket.io-stream for bi-directional
      // streaming, create a stream
      var stream = socket.createStream();
      // stream directly to server
      // it will be temp. stored locally
      socket.emit('stream', stream, {
        name: 'stream.wav',
        size: blob.size
      });
    }
  });
}

function audio_visualize(stream) {
  if (!audioCtx) {
    audioCtx = new AudioContext();
  }

  const source = audioCtx.createMediaStreamSource(stream);

  const analyser = audioCtx.createAnalyser();
  analyser.fftSize = 2048;
  const bufferLength = analyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);

  source.connect(analyser);
  //analyser.connect(audioCtx.destination);

  draw()

  function draw() {
    const WIDTH = canvas.width
    const HEIGHT = canvas.height;

    requestAnimationFrame(draw);

    analyser.getByteTimeDomainData(dataArray);

    canvasCtx.fillStyle = 'rgb(200, 200, 200)';
    canvasCtx.fillRect(0, 0, WIDTH, HEIGHT);

    canvasCtx.lineWidth = 2;
    canvasCtx.strokeStyle = 'rgb(0, 0, 0)';

    canvasCtx.beginPath();

    let sliceWidth = WIDTH * 1.0 / bufferLength;
    let x = 0;


    for (let i = 0; i < bufferLength; i++) {

      let v = dataArray[i] / 128.0;
      let y = v * HEIGHT / 2;

      if (i === 0) {
        canvasCtx.moveTo(x, y);
      } else {
        canvasCtx.lineTo(x, y);
      }

      x += sliceWidth;
    }

    canvasCtx.lineTo(canvas.width, canvas.height / 2);
    canvasCtx.stroke();

  }
}