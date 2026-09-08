"""Application entry point for AI Human Face Detector."""

from gui import FaceDetectorApp


def main() -> None:
    """Create and run the desktop application."""
    app = FaceDetectorApp()
    app.run()


if __name__ == "__main__":
    main()