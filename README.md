# AirCanvas

Real-time touchless drawing application powered by OpenCV and MediaPipe hand tracking.

AirCanvas allows users to draw in mid-air using hand gestures captured by a standard webcam. It includes fingertip tracking, an interactive color and eraser palette, and automatic geometric shape regularization.

---

## Features

- **Fingertip Drawing**: Tracks the index fingertip landmark to render smooth strokes on a persistent canvas layer.
- **✨ Magic Pencil (Turn Drawing into Real Object)**: Draw an outline in the air (e.g. banana, sunglasses, crown, apple, pizza) and watch it instantly materialize into a realistic transparent cutout object right where you drew it.
- **🎮 Play & Grab Objects in Mid-Air**: Pinch your index finger and thumb together near any materialized object to grab, drag, and move it across the room in real time.
- **Smart Tool Selection**: Switch between Cyan, Purple, Green, Amber, Magic, and Eraser via on-screen buttons, hand gestures, or web UI controls.
- **Automatic Shape Regularization**: Hand-drawn strokes resembling circles, rectangles, or lines automatically snap to clean geometric primitives.
- **Dual Perspective**: Natural selfie mirroring toggle ensures natural drawing motion.
- **Real-Time Web Interface**: Delivers 30 FPS multipart streaming in modern browsers with low latency.

---

## Installation

```bash
git clone https://github.com/deekshitvegi/aircanvas.git
cd aircanvas

pip install -r requirements.txt
```

---

## Usage

Start the web studio:

```bash
python main.py --port 2001
```

Open `http://localhost:2001` in your browser.

---

## Testing

```bash
pytest tests/ -v
```

---

## License

MIT License
