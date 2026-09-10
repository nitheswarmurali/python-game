"""Application entry point for AI Human Analysis & Emotion Detection."""

import logging
from pathlib import Path

from gui import FaceDetectorApp


def main() -> None:
    """Create and run the desktop application."""
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.FileHandler(output_dir / "application.log", encoding="utf-8"), logging.StreamHandler()],
    )
    app = FaceDetectorApp()
    app.run()


if __name__ == "__main__":
    main()