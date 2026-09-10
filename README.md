# AI Human Analysis & Emotion Detection System

A local Python desktop dashboard for real-time human analysis. The pipeline combines OpenCV face detection, optional DeepFace age/gender/emotion inference, MediaPipe full-body pose, MediaPipe hand landmarks, eye and smile heuristics, lightweight tracking, activity labels, screenshots, and analytics exports.

## Capabilities

- Multi-face detection with stable session IDs such as `#001`.
- DeepFace emotion probabilities for happy, neutral, sad, angry, fear, surprise, and disgust.
- DeepFace estimated age and gender probabilities.
- MediaPipe Pose skeleton with head, shoulders, arms, elbows, wrists, hips, knees, and ankles.
- MediaPipe Hands gesture labels: Open Palm, Fist, Pointing, Thumbs Up, and Victory Sign.
- Eye state, smile, head direction, attention score, and basic body activity labels.
- Dark Tkinter dashboard with webcam and image input.
- CSV and JSON analytics reports, annotated images, screenshots, and `output/application.log`.

The app uses lazy optional backends. It still starts in OpenCV fallback mode if MediaPipe or DeepFace is not installed, but model-backed pose, gesture, age, gender, and emotion fields will be marked unavailable.

## Installation

Use Python 3.11 as requested. On Windows, install Python with Tcl/Tk enabled, then run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The pinned stack is OpenCV 4.10.0.84, MediaPipe 0.10.14, TensorFlow 2.18.0, NumPy 2.1.1, Pillow 11.0.0, and CustomTkinter, with DeepFace installed from its latest release. DeepFace downloads its model weights on the first analysis run, so that first frame can be slower.

## Run

```powershell
python main.py
```

Click **Start Camera** or **Upload Image**. Use **Screenshot**, **Export CSV**, and **Export JSON** from the toolbar. Generated files are placed in `screenshots/`, `reports/`, and `output/`.

## Project structure

```text
AI_HUMAN_ANALYSIS/
├── main.py              # logging and application entry point
├── detector.py          # face detector, tracking, orchestration, result models
├── emotion.py           # lazy DeepFace emotion/age/gender service
├── age_gender.py        # age/gender compatibility service
├── pose.py              # MediaPipe Pose and body activity
├── gesture.py           # MediaPipe Hands and gesture labels
├── gui.py               # Tkinter desktop dashboard
├── utils.py             # folders, timestamps, JSON/CSV helpers
├── requirements.txt
├── models/
├── reports/
├── screenshots/
└── output/
```

## Notes on performance and accuracy

The engine throttles DeepFace analysis per tracked face and reuses the last result between model calls. Webcam throughput depends on camera resolution, CPU/GPU drivers, and model warm-up; 30 FPS is a target rather than a guarantee. Age, gender, emotion, attention, eye, smile, and activity values are estimates and should not be treated as identity, medical, employment, or safety decisions.

For GPU acceleration, install the TensorFlow build and GPU runtime appropriate for your platform before launching the app. OpenCV and MediaPipe will otherwise use CPU execution.

## Troubleshooting

- **Camera unavailable:** close other applications using the camera and check Windows camera privacy settings.
- **Pose or gestures unavailable:** verify that `mediapipe==0.10.14` is installed in the interpreter selected by VS Code.
- **Age, gender, or emotion unavailable:** verify `deepface` and `tensorflow==2.18.0`; the first successful call downloads model weights.
- **Slow startup:** DeepFace and TensorFlow initialization is model-heavy. Let the first analysis finish before judging steady-state FPS.
- **Tkinter error:** reinstall Python with Tcl/Tk and IDLE enabled.

## Online deployment

The browser UI deploys to Vercel, but the OpenCV and YOLO API must run on a Python host. Vercel Python functions are not suitable for this dependency stack because the model packages and runtime exceed typical serverless size and execution limits.

### 1. Deploy the API

Create a Render Web Service from this repository using `render.yaml`. After it is live, copy its HTTPS URL, for example `https://vision-forge-api.onrender.com`.

Set the API environment variable `CORS_ORIGIN` to the final Vercel URL. The first start downloads the YOLO checkpoint and can take several minutes. Files in `output/` are temporary on hosted instances.

### 2. Deploy the frontend

Import this repository into Vercel. The project uses the included `vercel.json` and runs `npm run build` automatically. Add this Vercel environment variable before deploying:

```text
VISION_API_URL=https://vision-forge-api.onrender.com
```

After deployment, the Vercel URL works on desktop and mobile. Camera access requires HTTPS and browser permission. Uploads and camera frames are sent to the API for processing.

For local development, leave `VISION_API_URL` unset and run `python webapp.py`; the frontend then calls the local Flask server.
