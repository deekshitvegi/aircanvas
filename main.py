"""
AirCanvas Launcher: Run in web mode (browser 30 FPS) or desktop mode (native OpenCV).
"""

import argparse
from src.server import run


def main():
    parser = argparse.ArgumentParser(description="AirCanvas - Touchless Gesture Drawing")
    parser.add_argument("--port", type=int, default=2001, help="Port for web studio (default: 2001)")
    args = parser.parse_args()
    run(port=args.port)


if __name__ == "__main__":
    main()
