# AI Human Face Detector

A production-minded desktop face detection dashboard built with Python, OpenCV, Pillow, and Tkinter. It supports webcam streams, images, and MP4/AVI/MOV video files with CPU-friendly Haar cascade detection.

## Features

- Real-time multi-face webcam detection with green boxes, confidence estimates, face count, and FPS.
- Image upload with automatic annotated output in `output/`.
- Frame-by-frame video processing on a worker thread, preserving a responsive GUI.
- Screenshot/result saving, detection history, and CSV report export.
- Dark dashboard sized at 1200x800 with camera and processing status.
- Clear errors for missing cameras, invalid media, missing models, and write failures.

## Installation

Requires Python 3.11.9 (Python 3.11 is recommended) and a working Tk installation. Tkinter is included with the standard Windows Python installer.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, install Python from python.org with **tcl/tk and IDLE** enabled. `tkinter` is part of Python and should not be installed with pip.

## Run

```powershell
python main.py
```

Use **Start Camera**, **Upload Image**, or **Upload Video**. Processed media is written to `output/`; source folders are available as `images/` and `videos/`.

## Model

The application first looks for `models/haarcascade_frontalface_default.xml`. If it is not present, it uses the copy packaged with `opencv-python`. This makes a fresh install work immediately while still allowing a project-local model to be supplied.

## Troubleshooting

- **Camera unavailable:** close other apps using the webcam, check Windows camera privacy permissions, and try another USB camera.
- **Invalid image/video:** use a readable JPG, PNG, BMP, WEBP, MP4, AVI, MOV, or MKV file.
- **Model missing:** reinstall `opencv-python` or place `haarcascade_frontalface_default.xml` in `models/`.
- **Permission error:** choose a writable output location and ensure the project folder is not read-only.
- **Low FPS:** reduce camera resolution, close other CPU-heavy applications, or increase `minNeighbors` in `gui.py` when constructing `FaceDetector`.

## Project structure

```text
human_face_detector/
├── main.py
├── detector.py
├── gui.py
├── utils.py
├── requirements.txt
├── models/
├── images/
├── videos/
└── output/
```