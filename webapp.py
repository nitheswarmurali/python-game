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
detection_lock = threading.Lock()


@app.after_request
def add_cors_headers(response):
    """Allow the Vercel frontend to call this API when hosted separately."""
    if allowed_origin:
        response.headers["Access-Control-Allow-Origin"] = allowed_origin
        response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


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
    }


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