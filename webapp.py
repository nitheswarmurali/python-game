"""Browser-based hybrid face and person detection application."""

from __future__ import annotations

import base64
import csv
import io
import ipaddress
import os
import socket
import threading
import time
import uuid
from urllib.parse import urlsplit
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file

from detector import HybridDetector
from utils import ensure_directories, timestamp

ROOT = Path(__file__).resolve().parent
ensure_directories(ROOT)
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024
allowed_origin = os.getenv("CORS_ORIGIN", "")
detector = HybridDetector(os.getenv("YOLO_MODEL", "yolo26s.pt"))
history: list[dict[str, object]] = []
face_tracks: dict[str, list[dict[str, object]]] = {}
detection_lock = threading.Lock()


@app.after_request
def add_cors_headers(response):
    """Allow the Vercel frontend to call this API when hosted separately."""
    if allowed_origin:
        response.headers["Access-Control-Allow-Origin"] = allowed_origin
        response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


def _face_signature(frame: np.ndarray, face: tuple[int, int, int, int, float]) -> np.ndarray | None:
    """Create a small normalized signature for matching a face between frames."""
    x, y, width, height, _ = face
    height_limit, width_limit = frame.shape[:2]
    crop = frame[max(0, y):min(height_limit, y + height), max(0, x):min(width_limit, x + width)]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    return cv2.normalize(gray, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)


def _capture_new_faces(frame: np.ndarray, faces: list[tuple[int, int, int, int, float]], track_id: str) -> list[dict[str, str]]:
    """Save one cropped photo for each face not seen in this camera session."""
    tracks = face_tracks.setdefault(track_id, [])
    captures: list[dict[str, str]] = []
    for face in faces:
        signature = _face_signature(frame, face)
        if signature is None:
            continue
        is_known = any(float(cv2.norm(signature, previous, cv2.NORM_L2)) < 8.0 for previous in tracks)
        if is_known:
            continue
        x, y, width, height, _ = face
        margin = int(max(width, height) * 0.2)
        y1, y2 = max(0, y - margin), min(frame.shape[0], y + height + margin)
        x1, x2 = max(0, x - margin), min(frame.shape[1], x + width + margin)
        crop = frame[y1:y2, x1:x2]
        name = f"face_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.jpg"
        path = ROOT / "output" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), crop) or not path.is_file():
            LOGGER.warning("Unable to save face capture: %s", path)
            continue
        tracks.append(signature)
        captures.append({"name": name, "url": f"/api/captures/{name}"})
    del tracks[:-50]
    return captures


def _download_image(url: str) -> bytes:
    """Download a public image URL without allowing local-network access."""
    parsed = urlsplit(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Paste a direct public image URL beginning with http:// or https://.")
    try:
        addresses = socket.getaddrinfo(parsed.hostname, None)
        if any(ipaddress.ip_address(address[4][0]).is_private for address in addresses):
            raise ValueError("Local and private-network URLs are not allowed.")
    except socket.gaierror as error:
        raise ValueError("The image URL host could not be found.") from error
    request = Request(
        url,
        headers={
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 VisionForge/1.0",
        },
    )
    try:
        with urlopen(request, timeout=15) as response:
            content_type = response.headers.get_content_type()
            if content_type not in {"image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp", "image/avif"}:
                raise ValueError("The URL returned a webpage or unsupported file. Use a direct JPG, PNG, WEBP, GIF, or BMP image URL.")
            payload = response.read(app.config["MAX_CONTENT_LENGTH"] + 1)
    except HTTPError as error:
        raise ValueError(f"The image host rejected the request (HTTP {error.code}). Try another direct image URL.") from error
    except URLError as error:
        raise ValueError("The image URL could not be reached. Check the URL and your internet connection.") from error
    if len(payload) > app.config["MAX_CONTENT_LENGTH"]:
        raise ValueError("The remote image is larger than 12 MB.")
    return payload


def _process(payload: bytes, use_yolo: bool, save_output: bool = True) -> dict[str, object]:
    """Decode and serialize an image, optionally saving the annotated result."""
    frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("The uploaded file is not a readable image.")
    started = time.perf_counter()
    with detection_lock:
        result, objects = detector.detect(frame, use_yolo=use_yolo)
    elapsed = max(time.perf_counter() - started, 0.001)
    ok, encoded = cv2.imencode(".jpg", result.frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("The processed image could not be encoded.")
    output_name = ""
    if save_output:
        output_path = ROOT / "output" / f"web_{timestamp()}_{int(time.time() * 1000) % 1000}.jpg"
        cv2.imwrite(str(output_path), result.frame)
        output_name = output_path.name
    people_count = len(objects) or len(result.faces)
    event = {"source": "web", "faces": len(result.faces), "people": people_count, "fps": round(1 / elapsed, 1)}
    history.append({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), **event})
    del history[:-50]
    return {
        "image": "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii"),
        "faces": len(result.faces),
        "people": people_count,
        "fps": round(1 / elapsed, 1),
        "output": output_name,
        "captures": [],
    }


@app.get("/api/captures/<name>")
def capture(name: str):
    """Serve a newly captured face photo from the output folder."""
    safe_name = Path(name).name
    path = ROOT / "output" / safe_name
    if not path.is_file() or not safe_name.startswith("face_"):
        return jsonify({"error": "Capture not found."}), 404
    return send_file(path, mimetype="image/jpeg")


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def status():
    return jsonify({"haar": True, "yolo": detector.yolo_ready, "yolo_model": detector.yolo_model_name, "error": detector.yolo_error})


@app.post("/api/detect")
def detect():
    upload = request.files.get("file")
    try:
        if upload is not None and upload.filename:
            payload = upload.read()
        else:
            image_url = request.form.get("url", "")
            if not image_url:
                return jsonify({"error": "Choose an image or enter a public image URL."}), 400
            payload = _download_image(image_url)
        save_output = request.form.get("save", "true").lower() == "true"
        result = _process(payload, request.form.get("yolo", "true").lower() == "true", save_output)
        if request.form.get("capture", "false").lower() == "true":
            frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            tracked_id = request.form.get("track_id", "default")[:80]
            with detection_lock:
                detected, _ = detector.detect(frame, use_yolo=False)
            result["captures"] = _capture_new_faces(frame, detected.faces, tracked_id)
        return jsonify(result)
    except (ValueError, OSError, cv2.error) as error:
        return jsonify({"error": str(error)}), 400


@app.get("/api/report.csv")
def report():
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["timestamp", "source", "faces", "people", "fps"])
    writer.writeheader()
    writer.writerows(history)
    return send_file(io.BytesIO(output.getvalue().encode()), mimetype="text/csv", as_attachment=True, download_name="detection_report.csv")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)