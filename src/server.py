import os
import time
import threading
from typing import Optional
import cv2
import numpy as np
from flask import Flask, Response, render_template_string, request, jsonify

from .camera import VideoStream
from .canvas import AirCanvas

app = Flask(__name__)


class CanvasServerEngine:
    def __init__(self):
        self.lock = threading.Lock()
        self.source = 0
        self.mirror = True
        self.stream: Optional[VideoStream] = None
        self.canvas = AirCanvas()
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_telemetry = {}
        self.is_running = True

        self._start_stream()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _start_stream(self):
        if self.stream is not None:
            self.stream.release()
        self.stream = VideoStream(source=self.source)
        if str(self.source).lower() == "synthetic":
            self.mirror = False

    def set_source(self, src):
        with self.lock:
            self.source = int(src) if str(src).isdigit() else src
            self._start_stream()

    def set_mirror(self, val: bool):
        with self.lock:
            self.mirror = bool(val)

    def _loop(self):
        while self.is_running:
            if not self.stream or not self.stream.is_opened():
                time.sleep(0.05)
                continue

            ret, frame = self.stream.read()
            if not ret or frame is None:
                time.sleep(0.02)
                continue

            if self.mirror:
                frame = cv2.flip(frame, 1)

            with self.lock:
                out, tel = self.canvas.process_frame(frame)
                self.latest_frame = out
                self.latest_telemetry = tel

            time.sleep(0.005)

    def get_jpeg(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            ret, jpeg = cv2.imencode(".jpg", self.latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return jpeg.tobytes() if ret else None


engine = CanvasServerEngine()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AirCanvas - Touchless Gesture Drawing</title>
    <style>
        body { background-color: #0d1117; color: #c9d1d9; font-family: sans-serif; margin: 0; display: flex; height: 100vh; overflow: hidden; }
        aside { width: 300px; background-color: #161b22; padding: 20px; border-right: 1px solid #30363d; display: flex; flex-direction: column; gap: 14px; }
        main { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 20px; }
        .video-box { max-width: 960px; width: 100%; border: 1px solid #30363d; border-radius: 8px; overflow: hidden; }
        img { width: 100%; display: block; }
        .pill-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; }
        button { background-color: #21262d; border: 1px solid #30363d; color: #c9d1d9; padding: 8px; border-radius: 6px; cursor: pointer; }
        button:hover { background-color: #30363d; }
        button.active { border-color: #58a6ff; color: #58a6ff; font-weight: bold; }
        .action-btn { background-color: #238636; color: white; width: 100%; margin-top: 6px; }
        .action-btn:hover { background-color: #2ea043; }
    </style>
</head>
<body>
    <aside>
        <h2>AirCanvas</h2>
        <p style="font-size: 0.8rem; color: #8b949e;">Touchless AI Drawing & Shape Snapping</p>
        
        <div>
            <label style="font-size: 0.8rem; color: #8b949e;">Select Tool / Color</label>
            <div class="pill-grid" style="margin-top: 6px;">
                <button onclick="setTool('CYAN')" class="active tool-btn" id="btnCYAN">Cyan</button>
                <button onclick="setTool('PURPLE')" class="tool-btn" id="btnPURPLE">Purple</button>
                <button onclick="setTool('GREEN')" class="tool-btn" id="btnGREEN">Green</button>
                <button onclick="setTool('AMBER')" class="tool-btn" id="btnAMBER">Amber</button>
                <button onclick="setTool('RED')" class="tool-btn" id="btnRED">Red</button>
                <button onclick="setTool('ERASER')" class="tool-btn" id="btnERASER">Eraser</button>
            </div>
        </div>

        <div>
            <label style="font-size: 0.8rem; color: #8b949e;">Camera Input</label>
            <select onchange="setSource(this.value)" style="width: 100%; padding: 6px; background: #0d1117; color: white; border: 1px solid #30363d; border-radius: 6px; margin-top: 4px;">
                <option value="0">Webcam</option>
                <option value="synthetic">Synthetic Stream</option>
            </select>
        </div>

        <div style="display: flex; justify-content: space-between; align-items: center;">
            <label for="mirrorBox" style="font-size: 0.85rem; cursor: pointer;">Mirror Camera</label>
            <input type="checkbox" id="mirrorBox" checked onchange="setMirror(this.checked)" />
        </div>

        <button class="action-btn" onclick="clearCanvas()">Clear Canvas</button>
        <button class="action-btn" style="background-color: #1f6feb;" onclick="saveSnapshot()">Save Snapshot</button>
    </aside>

    <main>
        <div class="video-box">
            <img src="/video_feed" alt="AirCanvas Stream" />
        </div>
        <div style="margin-top: 12px; font-family: monospace; font-size: 0.9rem; color: #3fb950;" id="statusBadge">
            30.0 FPS | TOOL: CYAN | MODE: HOVER
        </div>
    </main>

    <script>
        async function setTool(tool) {
            document.querySelectorAll(".tool-btn").forEach(b => b.classList.toggle("active", b.id === "btn" + tool));
            await fetch("/api/tool", { method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({ tool: tool }) });
        }
        async function setSource(src) {
            await fetch("/api/source", { method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({ source: src }) });
        }
        async function setMirror(val) {
            await fetch("/api/mirror", { method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({ mirror: val }) });
        }
        async function clearCanvas() {
            await fetch("/api/clear", { method: "POST" });
        }
        async function saveSnapshot() {
            const res = await fetch("/api/snapshot", { method: "POST" });
            const data = await res.json();
            alert("Snapshot saved: " + data.filename);
        }
        setInterval(async () => {
            try {
                const res = await fetch("/api/status");
                const data = await res.json();
                document.getElementById("statusBadge").innerText = `${data.fps} FPS | TOOL: ${data.active_tool} | MODE: ${data.mode}` + (data.snapped_shape ? ` | SNAPPED: ${data.snapped_shape.toUpperCase()}` : "");
                document.querySelectorAll(".tool-btn").forEach(b => b.classList.toggle("active", b.id === "btn" + data.active_tool));
            } catch (e) {}
        }, 200);
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


def gen():
    while True:
        b = engine.get_jpeg()
        if b:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + b + b"\r\n")
        time.sleep(0.03)


@app.route("/video_feed")
def video_feed():
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def get_status():
    with engine.lock:
        return jsonify(engine.latest_telemetry)


@app.route("/api/tool", methods=["POST"])
def set_tool_endpoint():
    t = request.get_json(force=True).get("tool", "CYAN")
    with engine.lock:
        engine.canvas.set_tool(t)
    return jsonify({"status": "ok"})


@app.route("/api/source", methods=["POST"])
def set_source_endpoint():
    s = request.get_json(force=True).get("source", 0)
    engine.set_source(s)
    return jsonify({"status": "ok"})


@app.route("/api/mirror", methods=["POST"])
def set_mirror_endpoint():
    m = request.get_json(force=True).get("mirror", True)
    engine.set_mirror(m)
    return jsonify({"status": "ok"})


@app.route("/api/clear", methods=["POST"])
def clear_endpoint():
    with engine.lock:
        engine.canvas.reset()
    return jsonify({"status": "ok"})


@app.route("/api/snapshot", methods=["POST"])
def snapshot_endpoint():
    os.makedirs("captures", exist_ok=True)
    with engine.lock:
        if engine.latest_frame is not None:
            ts = int(time.time())
            fn = os.path.join("captures", f"canvas_{ts}.png")
            cv2.imwrite(fn, engine.latest_frame)
            return jsonify({"status": "ok", "filename": fn})
    return jsonify({"status": "error"}), 400


def run(port: int = 2001):
    print(f"[AirCanvas] Running on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, threaded=True)


if __name__ == "__main__":
    run(2001)
