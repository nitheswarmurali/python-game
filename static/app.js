const fileInput = document.querySelector('#file-input');
const urlInput = document.querySelector('#url-input');
const urlButton = document.querySelector('#url-button');
const cameraButton = document.querySelector('#camera-button');
const stopButton = document.querySelector('#stop-button');
const fpsSelect = document.querySelector('#fps-select');
const yoloToggle = document.querySelector('#yolo-toggle');
const video = document.querySelector('#camera-video');
const canvas = document.querySelector('#capture-canvas');
const apiBase = (window.VISION_API_URL || '').replace(/\/$/, '');
const apiUrl = (path) => `${apiBase}${path}`;
let stream = null;
let cameraTimer = null;
let cameraRunning = false;
let busy = false;
const activity = (text) => {
  const item = document.createElement('li');
  item.innerHTML = '<span></span>' + text;
  document.querySelector('#activity-list').prepend(item);
  while (document.querySelector('#activity-list').children.length > 4) document.querySelector('#activity-list').lastElementChild.remove();
};

const render = (data, source) => {
  const image = document.querySelector('#result-image');
  image.src = data.image;
  image.style.display = 'block';
  document.querySelector('#empty-state').style.display = 'none';
  document.querySelector('#source-name').textContent = source;
  document.querySelector('#frame-state').textContent = 'ANALYZED';
  document.querySelector('#faces-count').textContent = data.faces;
  document.querySelector('#people-count').textContent = data.people;
  document.querySelector('#fps-count').textContent = data.fps;
  document.querySelector('#output-name').textContent = data.output || 'Camera frame not saved';
  const download = document.querySelector('#download-link');
  if (data.output) {
    download.href = data.image;
    download.classList.remove('disabled');
  } else {
    download.classList.add('disabled');
  }
};

const analyze = async (blob, source, saveOutput = true) => {
  if (busy) return;
  busy = true;
  const form = new FormData();
  form.append('file', blob, 'frame.jpg');
  form.append('yolo', yoloToggle.checked);
  form.append('save', saveOutput);
  try {
    const response = await fetch(apiUrl('/api/detect'), {method: 'POST', body: form});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Detection failed');
    render(data, source);
  } catch (error) { activity(error.message); }
  busy = false;
};

fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  if (file) { analyze(file, file.name, true); activity('Image analyzed'); }
});

urlButton.addEventListener('click', async () => {
  const url = urlInput.value.trim();
  if (!url) return activity('Paste an image URL first');
  const form = new FormData();
  form.append('url', url);
  form.append('yolo', yoloToggle.checked);
  form.append('save', true);
  urlButton.disabled = true;
  try {
    const response = await fetch(apiUrl('/api/detect'), {method: 'POST', body: form});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'URL detection failed');
    render(data, 'Remote image');
    activity('Public image URL analyzed');
  } catch (error) { activity(error.message); }
  urlButton.disabled = false;
});

cameraButton.addEventListener('click', async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({video: {facingMode: 'user'}, audio: false});
    video.srcObject = stream;
    cameraButton.disabled = true;
    stopButton.disabled = false;
    document.querySelector('#frame-state').textContent = 'LIVE';
    activity('Camera connected; frames are not saved');
    cameraRunning = true;
    scheduleFrame();
  } catch (error) { activity('Camera permission required'); }
});

const scheduleFrame = () => {
  if (!cameraRunning) return;
  const interval = 1000 / Number(fpsSelect.value);
  cameraTimer = setTimeout(() => {
    if (!video.videoWidth) return scheduleFrame();
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    canvas.toBlob(async blob => {
      await analyze(blob, 'Browser camera', false);
      scheduleFrame();
    }, 'image/jpeg', .86);
  }, interval);
};

stopButton.addEventListener('click', () => {
  cameraRunning = false;
  clearTimeout(cameraTimer);
  if (stream) stream.getTracks().forEach(track => track.stop());
  stream = null;
  cameraButton.disabled = false;
  stopButton.disabled = true;
  document.querySelector('#frame-state').textContent = 'PAUSED';
  activity('Camera stopped');
});

const reportLink = document.querySelector('#report-link');
if (reportLink) reportLink.href = apiUrl('/api/report.csv');

fetch(apiUrl('/api/status')).then(response => response.json()).then(status => {
  const system = document.querySelector('#system-status');
  const caption = document.querySelector('#yolo-caption');
  if (status.yolo) { system.textContent = 'Models online'; caption.textContent = `Free ${status.yolo_model} ready`; }
  else { system.textContent = 'Haar online / YOLO unavailable'; caption.textContent = 'Install ultralytics to enable'; yoloToggle.checked = false; }
}).catch(() => { document.querySelector('#system-status').textContent = 'Backend offline'; });