URL = window.URL || window.webkitURL;

var gumStream; 						//stream from getUserMedia()
var rec; 							//Recorder.js object
var input; 							//MediaStreamAudioSourceNode we'll be recording

var AudioContext = window.AudioContext || window.webkitAudioContext;
var audioContext //audio context to help us record
var recordButton = document.getElementById("recordButton");
var stopButton = document.getElementById("stopButton");
var pauseButton = document.getElementById("pauseButton");

recordButton.addEventListener("click", startRecording);
stopButton.addEventListener("click", stopRecording);
pauseButton.addEventListener("click", pauseRecording);

function uuidv4() {
	return ([1e7]+-1e3+-4e3+-8e3+-1e11).replace(/[018]/g, c =>
	  (c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c / 4).toString(16)
	);
  }
session_uuid = uuidv4();

function startRecording() {
	console.log("recordButton clicked");

    var constraints = { audio: true, video:false }
	recordButton.disabled = true;
	stopButton.disabled = false;
	pauseButton.disabled = false

	navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
		console.log("getUserMedia() success, stream created, initializing Recorder.js ...");
		audioContext = new AudioContext();
		gumStream = stream;
		input = audioContext.createMediaStreamSource(stream);
		rec = new Recorder(input,{numChannels:1})
		rec.record()
		console.log("Recording started");

		
		var options = {
			source: input,
			voice_stop: function() {stopRecording();}, 
			voice_start: function() {startRecording();}
		   }; 

	}).catch(function(err) {
    	recordButton.disabled = false;
    	stopButton.disabled = true;
    	pauseButton.disabled = true
	});

}

function pauseRecording(){
	console.log("pauseButton clicked rec.recording=",rec.recording );
	if (rec.recording){
		rec.stop();
		pauseButton.innerHTML="Resume";
	}else{
		rec.record()
		pauseButton.innerHTML="Pause";

	}
}

function stopRecording() {
	
	console.log("stopButton clicked");

	stopButton.disabled = true;
	recordButton.disabled = false;
	pauseButton.disabled = true;

	pauseButton.innerHTML="Pause";
	
	rec.stop();

	gumStream.getAudioTracks()[0].stop();

	rec.exportWAV(createDownloadLink);


}

var all_Client_transcripts = []
var all_Sony_replies = []

var last_reply_db_id = 0

function createDownloadLink(blob) {
	blob_uuid = uuidv4();
	console.log('blob')
	console.log(blob)
	var url = URL.createObjectURL(blob);
	console.log('url')
	console.log(url)
	var au = document.createElement('audio');
	var li = document.createElement('li');
	var link = document.createElement('a');
	var transcript = document.createElement('p');
	transcript.setAttribute("id", 'transcript_' + blob_uuid);
	var reply = document.createElement('p');
	reply.setAttribute("id", 'reply_' + blob_uuid);

	var filename = new Date().toISOString();

	au.controls = true;
	au.src = url;

	link.href = url;
	link.download = filename+".wav"; //download forces the browser to donwload the file using the  filename
	link.innerHTML = "Save to disk";

	var upload = document.createElement('a');
	upload.href="#";
	upload.innerHTML = "Upload";
	upload.addEventListener("click", upload)
	li.appendChild(transcript)
	li.appendChild(reply)

	recordingsList.appendChild(li);

	var xhr=new XMLHttpRequest();
	xhr.onload=function(e) {
		if(this.readyState === 4) {
			console.log("Server returned: ",e.target.responseText);
			response_dict = JSON.parse(e.target.responseText)
			console.log(response_dict)
			last_reply_db_id = response_dict['reply_db_id']
			transcript_text = response_dict['transcript']  
			reply_text = response_dict['reply']
			document.getElementById('transcript_' + blob_uuid).innerHTML = 'User: ' + transcript_text
			document.getElementById('reply_' + blob_uuid).innerHTML = 'AI: ' + reply_text.replace('\n', '<br>')
			var msg = new SpeechSynthesisUtterance();
			var voices = window.speechSynthesis.getVoices();
			msg.voice = voices.filter(function(voice) { return voice.name == 'Google US English'; })[0];
			msg.volume = 1; // From 0 to 1
			msg.rate = 1; // From 0.1 to 10
			msg.pitch = 1; // From 0 to 2
			msg.text = reply_text;
			msg.lang = 'en';
			speechSynthesis.speak(msg);
			
			all_Client_transcripts.push(transcript_text)
			all_Sony_replies.push(reply_text)
		}
	};
	var reader = new FileReader();
	reader.readAsDataURL(blob); 
	reader.onloadend = function() {
		var base64data = reader.result;                
		var fd=new FormData();
		fd.append("base64data",base64data);
		fd.append("uuid", blob_uuid);
		fd.append("session_uuid", session_uuid);
		fd.append("last_reply_db_id", last_reply_db_id);
		fd.append('all_Client_transcripts', JSON.stringify(all_Client_transcripts))
		fd.append('all_Sony_replies', JSON.stringify(all_Sony_replies))
		xhr.open("POST","./upload_audio",true);
		xhr.send(fd);
	}
}