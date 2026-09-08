"""OpenCV face detection services used by the desktop application."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)
FaceBox = Tuple[int, int, int, int, float]


@dataclass(frozen=True)
class DetectionResult:
    """A processed frame and the faces found in it."""

    frame: np.ndarray
    faces: List[FaceBox]


class FaceDetector:
    """Detect human faces with a CPU-friendly Haar cascade."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (40, 40),
    ) -> None:
        """Load the local model, falling back to OpenCV's packaged model."""
        self.scale_factor = max(1.01, scale_factor)
        self.min_neighbors = max(1, min_neighbors)
        self.min_size = min_size
        self.model_path = self._resolve_model(model_path)
        self.classifier = cv2.CascadeClassifier(str(self.model_path))
        if self.classifier.empty():
            raise FileNotFoundError(f"Unable to load face model: {self.model_path}")
        LOGGER.info("Loaded face model from %s", self.model_path)

    @staticmethod
    def _resolve_model(model_path: str | Path | None) -> Path:
        """Resolve a requested model or OpenCV's bundled Haar model."""
        requested = Path(model_path) if model_path else Path(__file__).parent / "models" / "haarcascade_frontalface_default.xml"
        if requested.is_file():
            return requested
        bundled = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if bundled.is_file():
            LOGGER.warning("Project model missing; using bundled model at %s", bundled)
            return bundled
        raise FileNotFoundError(
            "Missing haarcascade_frontalface_default.xml. Install opencv-python or add it to models/."
        )

    def detect(self, frame: np.ndarray, draw: bool = True) -> DetectionResult:
        """Detect faces in BGR image data and optionally draw annotations."""
        if frame is None or frame.size == 0:
            raise ValueError("The input frame is empty.")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        boxes, _, weights = self.classifier.detectMultiScale3(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
            outputRejectLevels=True,
        )
        faces: List[FaceBox] = []
        for index, (x, y, width, height) in enumerate(boxes):
            weight = float(weights[index]) if index < len(weights) else 0.0
            confidence = min(99.9, max(50.0, 50.0 + weight * 5.0))
            faces.append((int(x), int(y), int(width), int(height), confidence))
            if draw:
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 136), 2)
                label = f"Face {confidence:.0f}%"
                cv2.putText(frame, label, (x, max(22, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 136), 2)
        if draw:
            cv2.putText(
                frame,
                f"Detected Faces: {len(faces)}",
                (18, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 191, 255),
                2,
            )
        return DetectionResult(frame=frame, faces=faces)

    def detect_file(self, input_path: str | Path, output_path: str | Path) -> DetectionResult:
        """Read an image, annotate it, and save the result."""
        image = cv2.imread(os.fspath(input_path))
        if image is None:
            raise ValueError(f"Invalid or unreadable image: {input_path}")
        result = self.detect(image)
        if not cv2.imwrite(os.fspath(output_path), result.frame):
            raise PermissionError(f"Could not write image: {output_path}")
        return result