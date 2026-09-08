"""Dark Tkinter dashboard for webcam, image, and video face detection."""

from __future__ import annotations

import logging
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import cv2
from PIL import Image, ImageTk

from detector import FaceDetector
from utils import ensure_directories, export_report, timestamp

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent


class FaceDetectorApp:
    """Manage the application window and non-blocking media workflows."""

    def __init__(self) -> None:
        """Build the dashboard and initialize application state."""
        ensure_directories(ROOT)
        self.detector = FaceDetector()
        self.window = tk.Tk()
        self.window.title("AI Human Face Detector")
        self.window.geometry("1200x800")
        self.window.minsize(960, 640)
        self.window.configure(bg="#0A0A0A")
        self.capture: Optional[cv2.VideoCapture] = None
        self.camera_running = False
        self.video_running = False
        self.current_frame = None
        self.last_result = None
        self.last_faces = 0
        self.history: list[dict[str, object]] = []
        self.last_tick = time.perf_counter()
        self._setup_style()
        self._build_ui()
        self.window.protocol("WM_DELETE_WINDOW", self.close)

    def _setup_style(self) -> None:
        """Configure the restrained dark theme used by ttk controls."""
        style = ttk.Style(self.window)
        style.theme_use("clam") 
        style.configure("TButton", background="#171A1D", foreground="#E8F1F2", borderwidth=0, padding=(14, 10), font=("Segoe UI", 10, "bold"))
        style.map("TButton", background=[("active", "#00BFFF"), ("disabled", "#25282A")], foreground=[("active", "#061014")])
        style.configure("TLabel", background="#0A0A0A", foreground="#D7E3E5", font=("Segoe UI", 10))
        style.configure("Header.TLabel", foreground="#FFFFFF", font=("Segoe UI Semibold", 22))
        style.configure("Metric.TLabel", foreground="#00FF88", font=("Segoe UI Semibold", 14))

    def _build_ui(self) -> None:
        """Create navigation, preview, metrics, and activity panels."""
        header = tk.Frame(self.window, bg="#0A0A0A")
        header.pack(fill="x", padx=28, pady=(24, 12))
        ttk.Label(header, text="AI HUMAN FACE DETECTOR", style="Header.TLabel").pack(side="left")
        ttk.Label(header, text="OPENCV / CPU MODE", foreground="#00BFFF").pack(side="right", pady=6)

        toolbar = tk.Frame(self.window, bg="#111315")
        toolbar.pack(fill="x", padx=28, pady=(0, 16))
        for text, command in (("Start Camera", self.start_camera), ("Stop Camera", self.stop_camera), ("Upload Image", self.upload_image), ("Upload Video", self.upload_video), ("Save Result", self.save_result), ("Export Report", self.export_history)):
            ttk.Button(toolbar, text=text, command=command).pack(side="left", padx=6, pady=10)

        content = tk.Frame(self.window, bg="#0A0A0A")
        content.pack(fill="both", expand=True, padx=28, pady=(0, 22))
        preview_panel = tk.Frame(content, bg="#111315", highlightbackground="#25292B", highlightthickness=1)
        preview_panel.pack(side="left", fill="both", expand=True)
        self.preview = tk.Label(preview_panel, text="Choose a source to begin", bg="#111315", fg="#647276", font=("Segoe UI", 16))
        self.preview.pack(fill="both", expand=True, padx=2, pady=2)

        side = tk.Frame(content, bg="#0A0A0A", width=280)
        side.pack(side="right", fill="y", padx=(18, 0))
        side.pack_propagate(False)
        ttk.Label(side, text="LIVE METRICS", foreground="#00BFFF", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(4, 18))
        self.face_metric = ttk.Label(side, text="Detected Faces: 0", style="Metric.TLabel")
        self.face_metric.pack(anchor="w", pady=8)
        self.fps_metric = ttk.Label(side, text="FPS: 0.0", style="Metric.TLabel")
        self.fps_metric.pack(anchor="w", pady=8)
        self.status_metric = ttk.Label(side, text="Camera Status: Idle", style="Metric.TLabel")
        self.status_metric.pack(anchor="w", pady=8)
        ttk.Separator(side).pack(fill="x", pady=22)
        ttk.Label(side, text="ACTIVITY", foreground="#00BFFF", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 10))
        self.activity = tk.Listbox(side, height=14, bg="#111315", fg="#AFC0C4", relief="flat", highlightthickness=0, font=("Consolas", 9))
        self.activity.pack(fill="both", expand=True)
        self._log_activity("Ready for input")

    def _log_activity(self, text: str) -> None:
        """Add a timestamped message to the activity feed."""
        self.activity.insert(tk.END, f"{time.strftime('%H:%M:%S')}  {text}")
        self.activity.see(tk.END)

    def _show_frame(self, frame, faces: int, fps: float = 0.0) -> None:
        """Render an OpenCV BGR frame into the Tk preview area."""
        self.current_frame = frame.copy()
        self.last_faces = faces
        self.face_metric.configure(text=f"Detected Faces: {faces}")
        self.fps_metric.configure(text=f"FPS: {fps:.1f}")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((850, 650), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self.preview.configure(image=photo, text="")
        self.preview.image = photo

    def start_camera(self) -> None:
        """Open the default webcam and start scheduled frame processing."""
        if self.camera_running:
            return
        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW if cv2.CAP_DSHOW else 0)
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            messagebox.showerror("Camera unavailable", "No accessible camera was found.")
            return
        self.camera_running = True
        self.video_running = False
        self.status_metric.configure(text="Camera Status: Active")
        self._log_activity("Camera started")
        self._camera_tick()

    def _camera_tick(self) -> None:
        """Read, process, and schedule the next webcam frame."""
        if not self.camera_running or self.capture is None:
            return
        ok, frame = self.capture.read()
        if ok:
            result = self.detector.detect(frame)
            now = time.perf_counter()
            fps = 1.0 / max(now - self.last_tick, 0.001)
            self.last_tick = now
            self.last_result = result.frame
            self._show_frame(result.frame, len(result.faces), fps)
        self.window.after(15, self._camera_tick)

    def stop_camera(self) -> None:
        """Stop any active camera or video capture and release the device."""
        self.camera_running = False
        self.video_running = False
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.status_metric.configure(text="Camera Status: Idle")
        self._log_activity("Capture stopped")

    def upload_image(self) -> None:
        """Process a selected image and save an annotated copy."""
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if not path:
            return
        self.stop_camera()
        output = ROOT / "output" / f"image_{timestamp()}.jpg"
        try:
            result = self.detector.detect_file(path, output)
            self.last_result = result.frame
            self._show_frame(result.frame, len(result.faces))
            self._record("image", len(result.faces), 0.0)
            self._log_activity(f"Image: {len(result.faces)} face(s)")
        except (ValueError, PermissionError, OSError) as error:
            messagebox.showerror("Image error", str(error))

    def upload_video(self) -> None:
        """Process an uploaded video on a worker thread and save its output."""
        path = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv")])
        if not path:
            return
        self.stop_camera()
        if self.video_running:
            return
        self.video_running = True
        self.status_metric.configure(text="Camera Status: Processing video")
        output = ROOT / "output" / f"video_{timestamp()}.mp4"
        threading.Thread(target=self._process_video, args=(Path(path), output), daemon=True).start()

    def _process_video(self, input_path: Path, output_path: Path) -> None:
        """Process video frames without blocking the Tk event loop."""
        reader = cv2.VideoCapture(str(input_path))
        if not reader.isOpened():
            self.window.after(0, lambda: messagebox.showerror("Video error", "Invalid or unreadable video."))
            self.video_running = False
            return
        width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        source_fps = reader.get(cv2.CAP_PROP_FPS) or 30.0
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), source_fps, (width, height))
        if not writer.isOpened():
            reader.release()
            self.video_running = False
            self.window.after(0, lambda: messagebox.showerror("Video error", "Could not create output video."))
            return
        total_faces = 0
        frames = 0
        try:
            while self.video_running:
                ok, frame = reader.read()
                if not ok:
                    break
                result = self.detector.detect(frame)
                writer.write(result.frame)
                total_faces += len(result.faces)
                frames += 1
                self.window.after(0, self._show_frame, result.frame, len(result.faces), source_fps)
        finally:
            reader.release()
            writer.release()
            self.video_running = False
        average = total_faces / frames if frames else 0
        self.window.after(0, self._video_complete, output_path, average)

    def _video_complete(self, output_path: Path, average: float) -> None:
        """Update the UI after video processing completes."""
        self.status_metric.configure(text="Camera Status: Idle")
        self._record("video", round(average, 2), 0.0)
        self._log_activity(f"Video saved: {output_path.name}")
        messagebox.showinfo("Video complete", f"Processed video saved to:\n{output_path}")

    def save_result(self) -> None:
        """Save the currently displayed annotated frame."""
        if self.current_frame is None:
            messagebox.showinfo("Nothing to save", "Process an image, video, or camera frame first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG image", "*.jpg"), ("PNG image", "*.png")])
        if path and not cv2.imwrite(path, self.current_frame):
            messagebox.showerror("Save error", "Could not save the selected file.")
        elif path:
            self._log_activity(f"Result saved: {Path(path).name}")

    def _record(self, source: str, faces: int, fps: float) -> None:
        """Append a compact event to the in-memory analytics history."""
        self.history.append({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "source": source, "faces": faces, "fps": fps})

    def export_history(self) -> None:
        """Export analytics as a CSV report."""
        path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile=f"detection_report_{timestamp()}.csv", filetypes=[("CSV report", "*.csv")])
        if not path:
            return
        try:
            export_report(Path(path), self.history)
            self._log_activity("Detection report exported")
        except ValueError as error:
            messagebox.showinfo("No report", str(error))
        except OSError as error:
            messagebox.showerror("Report error", str(error))

    def close(self) -> None:
        """Release resources and close the application."""
        self.stop_camera()
        self.window.destroy()

    def run(self) -> None:
        """Start Tkinter's event loop."""
        self.window.mainloop()