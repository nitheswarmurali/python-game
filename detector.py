"""OpenCV face detection services used by the desktop application."""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, List, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)
FaceBox = Tuple[int, int, int, int, float]
ObjectBox = Tuple[int, int, int, int, float, str]


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


class HybridDetector:
    """Combine Haar face detection with optional YOLO26 person detection."""

    def __init__(self, yolo_model: str = "yolo26s.pt") -> None:
        self.face_detector = FaceDetector()
        self.yolo_model_name = yolo_model
        self.yolo: Any | None = None
        self.yolo_error = ""
        self._load_yolo()

    def _load_yolo(self) -> None:
        """Load YOLO lazily enough to keep Haar available if it is unavailable."""
        try:
            from ultralytics import YOLO

            self.yolo = YOLO(self.yolo_model_name)
        except Exception as error:  # The web app remains useful without YOLO.
            self.yolo_error = str(error)
            LOGGER.warning("YOLO26 unavailable: %s", error)

    @property
    def yolo_ready(self) -> bool:
        """Whether the YOLO26 checkpoint loaded successfully."""
        return self.yolo is not None

    def detect(self, frame: np.ndarray, use_yolo: bool = True) -> tuple[DetectionResult, List[ObjectBox]]:
        """Annotate a frame with face boxes and, when available, person boxes."""
        result = self.face_detector.detect(frame.copy())
        objects: List[ObjectBox] = []
        if not use_yolo or self.yolo is None:
            return result, objects

        try:
            predictions = self.yolo.predict(result.frame, verbose=False, conf=0.35, imgsz=640)
            boxes = predictions[0].boxes
            names = predictions[0].names
            for index in range(len(boxes)):
                class_id = int(boxes.cls[index].item())
                label = str(names[class_id])
                if label != "person":
                    continue
                x1, y1, x2, y2 = (int(value) for value in boxes.xyxy[index].tolist())
                confidence = float(boxes.conf[index].item() * 100)
                objects.append((x1, y1, x2 - x1, y2 - y1, confidence, label))
                cv2.rectangle(result.frame, (x1, y1), (x2, y2), (255, 173, 51), 2)
                cv2.putText(
                    result.frame,
                    f"YOLO {label} {confidence:.0f}%",
                    (x1, max(22, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 173, 51),
                    2,
                )
        except Exception as error:
            self.yolo_error = str(error)
            LOGGER.warning("YOLO26 inference failed: %s", error)
        return result, objects


@dataclass
class PersonAnalysis:
    """All visible analytics for one tracked face."""

    person_id: int
    box: tuple[int, int, int, int]
    face_confidence: float
    age: int | None = None
    age_range: str = "Unknown"
    gender: dict[str, float] = field(default_factory=dict)
    emotion: dict[str, float] = field(default_factory=dict)
    eye_state: str = "Unknown"
    head_direction: str = "Unknown"
    attention_score: float = 0.0
    body_action: str = "Unavailable"
    hand_actions: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class AnalysisResult:
    """Annotated frame and JSON-serializable analytics."""

    frame: np.ndarray
    people: list[PersonAnalysis]
    fps: float = 0.0
    pose_ready: bool = False
    emotion_ready: bool = False

    def as_dict(self) -> dict[str, object]:
        return {"people": [person.as_dict() for person in self.people], "person_count": len(self.people), "fps": round(self.fps, 1), "pose_ready": self.pose_ready, "emotion_ready": self.emotion_ready}


class HumanAnalysisEngine:
    """Coordinate face, DeepFace, MediaPipe pose, and hand analysis."""

    def __init__(self) -> None:
        from emotion import DeepFaceAnalyzer
        from gesture import GestureAnalyzer
        from pose import PoseAnalyzer

        self.face_detector = FaceDetector()
        self.emotion = DeepFaceAnalyzer()
        self.pose = PoseAnalyzer()
        self.gesture = GestureAnalyzer()
        self.eye_detector = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_eye_tree_eyeglasses.xml"))
        self.smile_detector = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_smile.xml"))
        self.next_id = 1
        self.tracks: dict[int, tuple[float, float]] = {}
        self.cached: dict[int, dict[str, object]] = {}
        self.frame_index = 0

    def process(self, frame: np.ndarray, fps: float = 0.0) -> AnalysisResult:
        """Process one BGR frame and annotate it in place."""
        self.frame_index += 1
        result = self.face_detector.detect(frame, draw=False)
        pose_points, body_action = self.pose.process(result.frame)
        hand_actions = self.gesture.process(result.frame)
        people: list[PersonAnalysis] = []
        current_tracks: dict[int, tuple[float, float]] = {}
        for x, y, width, height, confidence in result.faces:
            center = (x + width / 2, y + height / 2)
            person_id = self._track(center)
            current_tracks[person_id] = center
            crop = frame[max(0, y):y + height, max(0, x):x + width]
            cached = self.cached.get(person_id, {})
            if self.frame_index % 15 == 1 or not cached:
                cached = self.emotion.analyze(crop)
                self.cached[person_id] = cached
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.size else np.empty((0, 0), dtype=np.uint8)
            eyes = self.eye_detector.detectMultiScale(gray, 1.1, 5, minSize=(10, 10)) if gray.size and not self.eye_detector.empty() else []
            smiles = self.smile_detector.detectMultiScale(gray, 1.7, 20, minSize=(15, 15)) if gray.size and not self.smile_detector.empty() else []
            direction = self._head_direction(center, frame.shape[1], frame.shape[0])
            eye_state = "Open" if len(eyes) >= 1 else "Closed"
            attention = 92.0 if direction == "Center" and eye_state == "Open" else 55.0 if eye_state == "Open" else 20.0
            emotion = dict(cached.get("emotion", {}))
            if smiles and emotion:
                emotion["happy"] = max(float(emotion.get("happy", 0.0)), 75.0)
            person = PersonAnalysis(person_id, (x, y, width, height), confidence, cached.get("age"), self._age_range(cached.get("age")), dict(cached.get("gender", {})), emotion, eye_state, direction, attention, body_action, hand_actions)
            people.append(person)
            self._draw_person(result.frame, person, smiles)
        self.tracks = current_tracks
        cv2.putText(result.frame, f"People: {len(people)}  FPS: {fps:.1f}", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 255), 2)
        return AnalysisResult(result.frame, people, fps, self.pose.ready, self.emotion._deepface is not None)

    def _track(self, center: tuple[float, float]) -> int:
        for person_id, previous in self.tracks.items():
            if float(np.linalg.norm(np.subtract(center, previous))) < 90:
                return person_id
        person_id = self.next_id
        self.next_id += 1
        return person_id

    @staticmethod
    def _head_direction(center: tuple[float, float], width: int, height: int) -> str:
        horizontal = center[0] / max(width, 1)
        vertical = center[1] / max(height, 1)
        if horizontal < 0.35:
            return "Left"
        if horizontal > 0.65:
            return "Right"
        if vertical < 0.25:
            return "Up"
        if vertical > 0.75:
            return "Down"
        return "Center"

    @staticmethod
    def _age_range(age: object) -> str:
        if not isinstance(age, int):
            return "Unknown"
        if age <= 10:
            return "0-10"
        if age <= 20:
            return "11-20"
        if age <= 30:
            return "21-30"
        if age <= 40:
            return "31-40"
        if age <= 50:
            return "41-50"
        if age <= 60:
            return "51-60"
        return "60+"

    @staticmethod
    def _draw_person(frame: np.ndarray, person: PersonAnalysis, smiles: Any) -> None:
        x, y, width, height = person.box
        color = (0, 255, 136)
        cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
        label = f"#{person.person_id:03d} {person.age_range} | {person.head_direction} | {person.attention_score:.0f}%"
        cv2.putText(frame, label, (x, max(22, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)
        if smiles:
            cv2.putText(frame, "Smile", (x, y + height + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 2)