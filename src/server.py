import os
import time
import threading
from typing import Optional
import cv2
import numpy as np
from flask import Flask, Response, render_template_string, request, jsonify
from dotenv import load_dotenv

load_dotenv()

from .camera import VideoStream
from .canvas import AirCanvas

app = Flask(__name__)


class CanvasServerEngine:
    def __init__(self):
        self.lock = threading.Lock()
        self.source = 0
        self.mirror = True
        self.stream: Optional[VideoStream] = None
        # show_palette=False eliminates the ugly OpenCV top bar from the camera view
        self.canvas = AirCanvas(show_palette=False)
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_telemetry = {}
        self.is_running = True

        env_key = os.getenv("GEMINI_API_KEY")
        if env_key:
            self.canvas.set_api_key(env_key)

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
            ret, jpeg = cv2.imencode(".jpg", self.latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return jpeg.tobytes() if ret else None


engine = CanvasServerEngine()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AirCanvas Studio</title>
    <style>
        :root {
            --bg-canvas: #090a0f;
            --surface-1: #12151d;
            --surface-2: #1b1f2b;
            --surface-3: #242938;
            --border-subtle: #232838;
            --border-strong: #333a4f;
            --text-primary: #f0f3f8;
            --text-secondary: #909bb0;
            --accent-primary: #3b82f6;
            --radius-sm: 6px;
            --radius-md: 10px;
            --radius-lg: 14px;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, sans-serif;
            -webkit-font-smoothing: antialiased;
        }

        body {
            background-color: var(--bg-canvas);
            color: var(--text-primary);
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            user-select: none;
        }

        header {
            height: 52px;
            background-color: var(--surface-1);
            border-bottom: 1px solid var(--border-subtle);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 20px;
            flex-shrink: 0;
            z-index: 10;
        }

        .header-brand {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .brand-title {
            font-size: 0.95rem;
            font-weight: 600;
            color: #ffffff;
        }

        .brand-pill {
            font-size: 0.72rem;
            font-weight: 500;
            padding: 3px 8px;
            background-color: var(--surface-2);
            border: 1px solid var(--border-subtle);
            border-radius: 20px;
            color: var(--text-secondary);
        }

        .header-controls {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .status-tag {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 0.76rem;
            color: var(--text-secondary);
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace;
        }

        .status-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background-color: #10b981;
            box-shadow: 0 0 8px rgba(16, 185, 129, 0.6);
        }

        .ghost-button {
            background: transparent;
            border: 1px solid var(--border-subtle);
            color: var(--text-primary);
            padding: 6px 12px;
            border-radius: var(--radius-sm);
            font-size: 0.78rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.15s ease;
        }

        .ghost-button:hover {
            background-color: var(--surface-2);
            border-color: var(--border-strong);
        }

        .key-badge-unconfigured {
            background-color: #3b2020;
            border-color: #7f2323;
            color: #fca5a5;
        }

        .key-badge-connected {
            background-color: #132e22;
            border-color: #1d6d45;
            color: #86efac;
        }

        main {
            flex: 1;
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
            background: radial-gradient(circle at center, #11141f 0%, #08090d 100%);
            padding: 24px;
            overflow: hidden;
        }

        .canvas-frame {
            position: relative;
            max-width: 960px;
            width: 100%;
            aspect-ratio: 16 / 9;
            background-color: #000;
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-lg);
            overflow: hidden;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.6);
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .canvas-frame img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        /* Bottom Studio Dock */
        .bottom-dock {
            position: absolute;
            bottom: 24px;
            left: 50%;
            transform: translateX(-50%);
            background-color: rgba(18, 21, 29, 0.94);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--border-strong);
            border-radius: 40px;
            padding: 6px 14px;
            display: flex;
            align-items: center;
            gap: 8px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
            z-index: 20;
        }

        .dock-tool {
            background: transparent;
            border: none;
            color: var(--text-secondary);
            padding: 7px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.15s ease;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .dock-tool:hover {
            color: var(--text-primary);
            background-color: var(--surface-2);
        }

        .dock-tool.active {
            background-color: #2563eb;
            color: #ffffff;
            box-shadow: 0 0 14px rgba(37, 99, 235, 0.4);
        }

        .dock-divider {
            width: 1px;
            height: 20px;
            background-color: var(--border-strong);
        }

        .dock-color-swatch {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
        }

        /* Top-Right Generator Panel */
        .generator-panel {
            position: absolute;
            top: 24px;
            right: 24px;
            width: 320px;
            background-color: rgba(18, 21, 29, 0.94);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border-strong);
            border-radius: var(--radius-md);
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 12px;
            box-shadow: 0 12px 36px rgba(0, 0, 0, 0.5);
            z-index: 20;
        }

        .panel-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .panel-title {
            font-size: 0.82rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            color: var(--text-secondary);
        }

        .panel-input-wrap {
            display: flex;
            gap: 8px;
        }

        input[type="text"], input[type="password"] {
            flex: 1;
            background-color: var(--surface-2);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-sm);
            color: var(--text-primary);
            padding: 8px 10px;
            font-size: 0.82rem;
            outline: none;
        }

        input:focus {
            border-color: var(--accent-primary);
        }

        .btn-primary {
            background-color: #2563eb;
            color: #ffffff;
            border: none;
            padding: 8px 12px;
            border-radius: var(--radius-sm);
            font-size: 0.82rem;
            font-weight: 500;
            cursor: pointer;
            transition: background-color 0.15s;
        }

        .btn-primary:hover {
            background-color: #1d4ed8;
        }

        .btn-secondary {
            background-color: var(--surface-2);
            border: 1px solid var(--border-subtle);
            color: var(--text-primary);
            padding: 7px 10px;
            border-radius: var(--radius-sm);
            font-size: 0.78rem;
            cursor: pointer;
            transition: all 0.15s;
        }

        .btn-secondary:hover {
            background-color: var(--surface-3);
            border-color: var(--border-strong);
        }

        .chip-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 6px;
        }

        .chip {
            background-color: var(--surface-2);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-sm);
            color: var(--text-secondary);
            font-size: 0.74rem;
            padding: 6px 8px;
            text-align: center;
            cursor: pointer;
            transition: all 0.15s;
        }

        .chip:hover {
            color: var(--text-primary);
            border-color: var(--border-strong);
            background-color: var(--surface-3);
        }

        .instruction-note {
            font-size: 0.78rem;
            color: var(--text-secondary);
            line-height: 1.45;
            background-color: var(--surface-2);
            padding: 10px 12px;
            border-radius: var(--radius-sm);
            border-left: 3px solid var(--accent-primary);
        }

        /* Settings Modal */
        .modal-overlay {
            position: fixed;
            inset: 0;
            background: rgba(0, 0, 0, 0.75);
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 100;
        }

        .modal-card {
            width: 440px;
            background-color: var(--surface-1);
            border: 1px solid var(--border-strong);
            border-radius: var(--radius-lg);
            padding: 22px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
    </style>
</head>
<body>

    <header>
        <div class="header-brand">
            <span class="brand-title">AirCanvas Studio</span>
            <span class="brand-pill">Gesture Perception & Generative Objects</span>
        </div>

        <div class="header-controls">
            <div class="status-tag">
                <span class="status-dot"></span>
                <span id="fpsIndicator">30.0 FPS</span>
            </div>

            <button class="ghost-button" onclick="toggleMirrorMode()">
                Mirror: <span id="mirrorState">Selfie</span>
            </button>

            <button id="apiKeyBtn" class="ghost-button key-badge-unconfigured" onclick="openKeyModal()">
                Gemini API: <span id="apiKeyLabel">Connect Key</span>
            </button>
        </div>
    </header>

    <main>
        <div class="canvas-frame">
            <img src="/video_feed" alt="Studio Stream" />
        </div>

        <!-- Studio Generative Object Panel -->
        <div class="generator-panel">
            <div class="panel-header">
                <span class="panel-title">Nano Banana Generator</span>
                <span id="objectCounter" style="font-size: 0.72rem; color: var(--text-secondary); font-family: monospace;">0 Objects</span>
            </div>

            <div style="font-size: 0.78rem; color: var(--text-secondary); line-height: 1.35;">
                Enter any object to synthesize a realistic transparent cutout via Nano Banana:
            </div>

            <div class="panel-input-wrap">
                <input type="text" id="objectPrompt" placeholder="Object (e.g. airpods, cricket bat, guitar)" />
                <button class="btn-primary" id="createBtn" onclick="materializePrompt()">Create</button>
            </div>

            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 2px;">
                <span style="font-size: 0.72rem; color: var(--text-secondary); text-transform: uppercase;">Authentic Cutouts</span>
                <button class="btn-secondary" style="padding: 3px 8px; font-size: 0.7rem; color: #f87171;" onclick="clearObjects()">Clear Objects</button>
            </div>

            <div class="chip-grid">
                <div class="chip" onclick="spawnCutout('airpods')">AirPods</div>
                <div class="chip" onclick="spawnCutout('cricket bat')">Cricket Bat</div>
                <div class="chip" onclick="spawnCutout('guitar')">Guitar</div>
                <div class="chip" onclick="spawnCutout('sunglasses')">Sunglasses</div>
                <div class="chip" onclick="spawnCutout('banana')">Banana</div>
                <div class="chip" onclick="spawnCutout('crown')">Crown</div>
            </div>

            <div class="instruction-note">
                <strong>Mid-Air Grab:</strong> Pinch your index finger and thumb together near any object to pick it up and move it across your screen.
            </div>
        </div>

        <!-- Floating Studio Dock -->
        <div class="bottom-dock">
            <button class="dock-tool active" data-tool="CYAN" onclick="selectTool('CYAN')">
                <span class="dock-color-swatch" style="background: #00e5ff;"></span>
                Cyan
            </button>
            <button class="dock-tool" data-tool="PURPLE" onclick="selectTool('PURPLE')">
                <span class="dock-color-swatch" style="background: #ec4899;"></span>
                Purple
            </button>
            <button class="dock-tool" data-tool="GREEN" onclick="selectTool('GREEN')">
                <span class="dock-color-swatch" style="background: #10b981;"></span>
                Green
            </button>
            <button class="dock-tool" data-tool="AMBER" onclick="selectTool('AMBER')">
                <span class="dock-color-swatch" style="background: #f59e0b;"></span>
                Amber
            </button>
            <div class="dock-divider"></div>
            <button class="dock-tool" data-tool="MAGIC" onclick="selectTool('MAGIC')" style="color: #f472b6;">
                Magic Sketch
            </button>
            <button class="dock-tool" data-tool="ERASER" onclick="selectTool('ERASER')">
                Eraser
            </button>
            <div class="dock-divider"></div>
            <button class="dock-tool" onclick="clearStrokes()">
                Clear Lines
            </button>
            <button class="dock-tool" style="color: #f87171;" onclick="clearObjects()">
                Remove All Objects
            </button>
            <button class="dock-tool" onclick="saveSnapshot()">
                Snapshot
            </button>
        </div>
    </main>

    <!-- API Key Settings Modal -->
    <div id="keyModal" class="modal-overlay">
        <div class="modal-card">
            <div style="font-size: 1rem; font-weight: 600;">Connect Gemini API Key</div>
            <div style="font-size: 0.8rem; color: var(--text-secondary); line-height: 1.45;">
                Please paste your Google AI Studio Gemini API key below. This connects Google Imagen 3 (Nano Banana) to generate photorealistic AI cutouts from any prompt or drawing, and automatically persists to your local environment.
            </div>

            <input type="password" id="modalKeyInput" placeholder="AIzaSy..." />

            <div style="display: flex; justify-content: flex-end; gap: 8px;">
                <button class="btn-secondary" onclick="closeKeyModal()">Cancel</button>
                <button class="btn-primary" onclick="saveApiKey()">Save & Connect</button>
            </div>
        </div>
    </div>

    <script>
        let isMirror = true;

        async function selectTool(tool) {
            document.querySelectorAll(".dock-tool[data-tool]").forEach(b => {
                b.classList.toggle("active", b.getAttribute("data-tool") === tool);
            });
            await fetch("/api/tool", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ tool: tool })
            });
        }

        async function toggleMirrorMode() {
            isMirror = !isMirror;
            document.getElementById("mirrorState").innerText = isMirror ? "Selfie" : "Standard";
            await fetch("/api/mirror", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ mirror: isMirror })
            });
        }

        async function materializePrompt() {
            const prompt = document.getElementById("objectPrompt").value.trim();
            if (!prompt) return;
            const btn = document.getElementById("createBtn");
            const originalText = btn.innerText;
            btn.innerText = "Generating...";
            btn.disabled = true;

            try {
                const res = await fetch("/api/materialize", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ prompt: prompt })
                });
                const data = await res.json();
                document.getElementById("objectPrompt").value = "";
            } catch (e) {
                console.error("Error creating object:", e);
            } finally {
                btn.innerText = originalText;
                btn.disabled = false;
            }
        }

        async function spawnCutout(name) {
            await fetch("/api/spawn", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name: name })
            });
        }

        async function clearStrokes() {
            await fetch("/api/clear", { method: "POST" });
        }

        async function clearObjects() {
            await fetch("/api/clear_objects", { method: "POST" });
        }

        async function saveSnapshot() {
            const res = await fetch("/api/snapshot", { method: "POST" });
            const data = await res.json();
            if (data.status === "ok") {
                alert("Snapshot saved: " + data.filename);
            }
        }

        function openKeyModal() {
            document.getElementById("keyModal").style.display = "flex";
        }

        function closeKeyModal() {
            document.getElementById("keyModal").style.display = "none";
        }

        async function saveApiKey() {
            const key = document.getElementById("modalKeyInput").value.trim();
            if (!key) return;
            const res = await fetch("/api/set_api_key", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ key: key })
            });
            const data = await res.json();
            if (data.status === "ok") {
                const btn = document.getElementById("apiKeyBtn");
                btn.className = "ghost-button key-badge-connected";
                document.getElementById("apiKeyLabel").innerText = "Connected";
                closeKeyModal();
            }
        }

        setInterval(async () => {
            try {
                const res = await fetch("/api/status");
                const data = await res.json();
                document.getElementById("fpsIndicator").innerText = (data.fps || 30.0).toFixed(1) + " FPS";
                document.getElementById("objectCounter").innerText = `${data.objects_count || 0} Objects`;

                const btn = document.getElementById("apiKeyBtn");
                if (data.has_api_key) {
                    btn.className = "ghost-button key-badge-connected";
                    document.getElementById("apiKeyLabel").innerText = "Connected";
                } else {
                    btn.className = "ghost-button key-badge-unconfigured";
                    document.getElementById("apiKeyLabel").innerText = "Connect Key";
                }

                document.querySelectorAll(".dock-tool[data-tool]").forEach(b => {
                    b.classList.toggle("active", b.getAttribute("data-tool") === data.active_tool);
                });
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


@app.route("/api/clear_objects", methods=["POST"])
def clear_objects_endpoint():
    with engine.lock:
        engine.canvas.object_mgr.clear()
    return jsonify({"status": "ok"})


@app.route("/api/materialize", methods=["POST"])
def materialize_endpoint():
    data = request.get_json(force=True)
    prompt = data.get("prompt") or data.get("hint")
    with engine.lock:
        obj = engine.canvas.materialize_current(hint=prompt)
        name = obj.name if obj else "object"
    return jsonify({"status": "ok", "materialized": name})


@app.route("/api/spawn", methods=["POST"])
def spawn_endpoint():
    name = request.get_json(force=True).get("name", "apple")
    with engine.lock:
        engine.canvas.spawn_object(name)
    return jsonify({"status": "ok"})


@app.route("/api/set_api_key", methods=["POST"])
def set_api_key_endpoint():
    key = request.get_json(force=True).get("key", "").strip()
    with engine.lock:
        engine.canvas.set_api_key(key)
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
