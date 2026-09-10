"""Desktop dashboard for the AI Human Analysis System."""

from __future__ import annotations

import csv
import logging
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from detector import AnalysisResult, HumanAnalysisEngine
from utils import ensure_directories, export_json, timestamp

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent
try:
    import customtkinter as ctk

    ctk.set_appearance_mode("dark")
    CUSTOMTKINTER_READY = True
except ImportError:
    ctk = None
    CUSTOMTKINTER_READY = False


class FaceDetectorApp:
    """Own the Tk event loop and keep analysis work on scheduled camera ticks."""

    def __init__(self) -> None:
        ensure_directories(ROOT)
        self.engine = HumanAnalysisEngine()
        self.window = ctk.CTk() if CUSTOMTKINTER_READY and ctk is not None else tk.Tk()
        self.window.title("AI Human Analysis & Emotion Detection System")
        self.window.geometry("1420x860")
        self.window.minsize(1100, 700)
        self.window.configure(bg="#081014")
        self.capture: cv2.VideoCapture | None = None
        self.running = False
        self.current_frame = None
        self.last_result: AnalysisResult | None = None
        self.history: list[dict[str, object]] = []
        self.last_tick = time.perf_counter()
        self._setup_style()
        self._build_ui()
        self.window.protocol("WM_DELETE_WINDOW", self.close)

    def _setup_style(self) -> None:
        style = ttk.Style(self.window)
        style.theme_use("clam")
        style.configure("TButton", background="#152228", foreground="#E8F3F4", borderwidth=0, padding=(12, 9), font=("Segoe UI", 10, "bold"))
        style.map("TButton", background=[("active", "#00C2A8")], foreground=[("active", "#061014")])
        style.configure("TLabel", background="#081014", foreground="#D9E8E8", font=("Segoe UI", 10))
        style.configure("Header.TLabel", foreground="#F5FFFF", font=("Segoe UI Semibold", 23))
        style.configure("Metric.TLabel", foreground="#00E0B3", font=("Segoe UI Semibold", 13))

    def _build_ui(self) -> None:
        header = tk.Frame(self.window, bg="#081014")
        header.pack(fill="x", padx=28, pady=(22, 12))
        ttk.Label(header, text="AI HUMAN ANALYSIS", style="Header.TLabel").pack(side="left")
        ttk.Label(header, text="LOCAL VISION LAB  /  MEDIAPIPE + DEEPFACE", foreground="#00C2A8").pack(side="right", pady=7)
        toolbar = tk.Frame(self.window, bg="#101B20")
        toolbar.pack(fill="x", padx=28, pady=(0, 16))
        for label, command in (("Start Camera", self.start_camera), ("Stop", self.stop_camera), ("Upload Image", self.upload_image), ("Screenshot", self.save_screenshot), ("Export CSV", self.export_csv), ("Export JSON", self.export_json)):
            ttk.Button(toolbar, text=label, command=command).pack(side="left", padx=5, pady=9)
        content = tk.Frame(self.window, bg="#081014")
        content.pack(fill="both", expand=True, padx=28, pady=(0, 22))
        stage = tk.Frame(content, bg="#101B20", highlightbackground="#21333A", highlightthickness=1)
        stage.pack(side="left", fill="both", expand=True)
        self.preview = tk.Label(stage, text="Start the camera or upload an image", bg="#101B20", fg="#71878A", font=("Segoe UI", 16))
        self.preview.pack(fill="both", expand=True, padx=2, pady=2)
        side = tk.Frame(content, bg="#081014", width=330)
        side.pack(side="right", fill="y", padx=(18, 0))
        side.pack_propagate(False)
        ttk.Label(side, text="SYSTEM STATUS", foreground="#00C2A8", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(3, 12))
        self.status = ttk.Label(side, text="Idle", style="Metric.TLabel")
        self.status.pack(anchor="w", pady=4)
        self.fps = ttk.Label(side, text="FPS  0.0", style="Metric.TLabel")
        self.fps.pack(anchor="w", pady=4)
        self.people = ttk.Label(side, text="People  0", style="Metric.TLabel")
        self.people.pack(anchor="w", pady=4)
        ttk.Separator(side).pack(fill="x", pady=15)
        ttk.Label(side, text="PERSON ANALYTICS", foreground="#00C2A8", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 8))
        self.person_list = tk.Listbox(side, bg="#101B20", fg="#D9E8E8", selectbackground="#00C2A8", relief="flat", highlightthickness=0, font=("Consolas", 9), height=18)
        self.person_list.pack(fill="both", expand=True)
        self.activity = ttk.Label(side, text="Ready", foreground="#71878A", wraplength=310)
        self.activity.pack(anchor="w", pady=(12, 4))

    def _camera_tick(self) -> None:
        if not self.running or self.capture is None:
            return
        ok, frame = self.capture.read()
        if ok:
            now = time.perf_counter()
            result = self.engine.process(frame, 1.0 / max(now - self.last_tick, 0.001))
            self.last_tick = now
            self.last_result = result
            self._show_result(result)
            self._record("camera", result)
        self.window.after(15, self._camera_tick)

    def _show_result(self, result: AnalysisResult) -> None:
        self.current_frame = result.frame.copy()
        self.status.configure(text=f"Active  |  Pose {'ready' if result.pose_ready else 'fallback'}")
        self.fps.configure(text=f"FPS  {result.fps:.1f}")
        self.people.configure(text=f"People  {len(result.people)}")
        self.person_list.delete(0, tk.END)
        for person in result.people:
            emotion = max(person.emotion, key=person.emotion.get) if person.emotion else "Unavailable"
            gender = max(person.gender, key=person.gender.get) if person.gender else "Unavailable"
            self.person_list.insert(tk.END, f"#{person.person_id:03d}  {person.age_range}  {gender}")
            self.person_list.insert(tk.END, f"  {emotion}  |  {person.eye_state}  |  {person.head_direction}")
            self.person_list.insert(tk.END, f"  Attention {person.attention_score:.0f}%  |  {person.body_action}")
            if person.hand_actions:
                self.person_list.insert(tk.END, f"  Hands: {', '.join(person.hand_actions)}")
        rgb = cv2.cvtColor(result.frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((1000, 780), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self.preview.configure(image=photo, text="")
        self.preview.image = photo

    def start_camera(self) -> None:
        if self.running:
            return
        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            messagebox.showerror("Camera unavailable", "No accessible camera was found.")
            return
        self.running = True
        self.last_tick = time.perf_counter()
        self.activity.configure(text="Camera active. Analysis stays on this device.")
        self._camera_tick()

    def stop_camera(self) -> None:
        self.running = False
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.status.configure(text="Idle")

    def upload_image(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if not path:
            return
        self.stop_camera()
        frame = cv2.imread(path)
        if frame is None:
            messagebox.showerror("Image error", "The selected image could not be read.")
            return
        result = self.engine.process(frame)
        self.last_result = result
        self._show_result(result)
        output = ROOT / "output" / f"analysis_{timestamp()}.jpg"
        cv2.imwrite(str(output), result.frame)
        self.activity.configure(text=f"Analyzed {Path(path).name}; saved {output.name}")
        self._record("image", result)

    def save_screenshot(self) -> None:
        if self.current_frame is None:
            messagebox.showinfo("No frame", "Start the camera or analyze an image first.")
            return
        path = ROOT / "screenshots" / f"screenshot_{timestamp()}.jpg"
        cv2.imwrite(str(path), self.current_frame)
        self.activity.configure(text=f"Screenshot saved to {path.name}")

    def _record(self, source: str, result: AnalysisResult) -> None:
        event = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "source": source, "fps": round(result.fps, 1), **result.as_dict()}
        self.history.append(event)
        del self.history[:-300]

    def export_csv(self) -> None:
        if not self.history:
            messagebox.showinfo("No report", "Analyze at least one frame first.")
            return
        path = ROOT / "reports" / f"analytics_{timestamp()}.csv"
        with path.open("w", newline="", encoding="utf-8") as report:
            writer = csv.DictWriter(report, fieldnames=["timestamp", "source", "fps", "person_count", "people"])
            writer.writeheader()
            writer.writerows({key: event.get(key, "") for key in writer.fieldnames} for event in self.history)
        self.activity.configure(text=f"CSV report saved to {path.name}")

    def export_json(self) -> None:
        if not self.history:
            messagebox.showinfo("No report", "Analyze at least one frame first.")
            return
        path = ROOT / "reports" / f"analytics_{timestamp()}.json"
        export_json(path, self.history)
        self.activity.configure(text=f"JSON analytics saved to {path.name}")

    def close(self) -> None:
        self.stop_camera()
        self.window.destroy()

    def run(self) -> None:
        self.window.mainloop()
