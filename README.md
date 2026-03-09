# 🎭 Realtime Emotion Detector

<div align="center">

![Emotion Detector Banner](https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=Emotion%20Detector&fontSize=50&fontColor=fff&animation=twinkling&fontAlignY=35&desc=Real-time%20Facial%20Emotion%20Recognition%20%7C%20No%20API%20Key%20Required&descAlignY=55&descSize=16)

<p>
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-4.9-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/DeepFace-0.0.93-FF6F00?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/No%20API%20Key-100%25%20Local-brightgreen?style=for-the-badge&logo=lock&logoColor=white"/>
  <img src="https://img.shields.io/badge/Platform-Windows%20%7C%20Mac%20%7C%20Linux-blue?style=for-the-badge"/>
</p>

<p>
  <b>Detect human emotions in real-time directly from your webcam.</b><br/>
  Powered by DeepFace's pre-trained neural network — works offline, no cloud, no cost.
</p>

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [How It Works](#-how-it-works) • [Tech Stack](#-tech-stack) • [Roadmap](#-roadmap)

</div>

---

## 🌟 Why This Project?

Most emotion detection demos are either:
- ☁️ Cloud-based (expensive, slow, privacy risk)
- 🔑 Require paid API keys
- 🧱 Complex to set up

This project is **different** — it runs **100% on your machine**, detects emotions in **real time** at 20+ FPS, and installs in under 2 minutes. It's a great foundation for building smart mirrors, mood-based music players, mental health tools, and more.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎯 **7 Emotions Detected** | Happy, Sad, Angry, Surprise, Fear, Disgust, Neutral |
| 📊 **Live Confidence Bars** | See percentage confidence for every emotion alongside your face |
| 🧠 **Smoothed Predictions** | Rolling 5-frame history eliminates flickering and jitter |
| 📸 **Screenshot Mode** | Press `S` to capture and save the current frame |
| ⚡ **Optimised Performance** | Runs DeepFace every 3 frames — smooth video, accurate results |
| 🌈 **Color-coded Overlays** | Each emotion gets its own unique colour on the bounding box |
| 🔒 **100% Private** | No data ever leaves your machine |
| 🖥️ **Cross-platform** | Works on Windows, macOS, and Linux |

---

## 🖥️ Demo

```
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   😄  HAPPY                          FPS: 24.3              ║
║                                      Frames: 412            ║
║   ╭──────────────────╮               Saved: 2               ║
║   │                  │                                       ║
║   │   ░░░░░░░░░░░░   │  happy    ████████  87.3%            ║
║   │   ░  FACE  ░░░   │  neutral  ███       10.1%            ║
║   │   ░░░░░░░░░░░░   │  sad      █          2.1%            ║
║   │                  │  angry               0.3%            ║
║   ╰──────────────────╯  ...                                  ║
║                                                              ║
║  [S] Screenshot    [R] Reset    [Q/ESC] Quit                 ║
╚══════════════════════════════════════════════════════════════╝
```

> 📹 **Tip:** After running, press `S` to capture a screenshot and add it to this README for extra impact!

---

## 📦 Installation

### Prerequisites
- Python 3.9 or higher → [Download Python](https://www.python.org/downloads/)
- A working webcam
- ~500MB free disk space (for model weights, downloaded automatically on first run)

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/realtime-emotion-detector.git
cd realtime-emotion-detector
```

---

### Step 2 — Create a virtual environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **First-time setup:** DeepFace will automatically download the pre-trained VGG-Face model weights (~100MB) on the very first run. This only happens once — after that, everything runs fully offline.

---

### Step 4 — Run the app

```bash
python app.py
```

A window will open showing your live webcam feed with emotion overlays. That's it! 🎉

---

## 🎮 Controls

| Key | Action |
|-----|--------|
| `S` | 📸 Save a screenshot to the `/screenshots` folder |
| `R` | 🔄 Reset session stats and emotion history |
| `Q` or `ESC` | ❌ Quit the application |

---

## 🧠 How It Works

```
┌─────────────┐     ┌──────────────┐     ┌────────────────────┐     ┌─────────────────┐
│             │     │              │     │                    │     │                 │
│   Webcam    │────▶│    OpenCV    │────▶│     DeepFace       │────▶│  Overlay + Draw │
│  (30 FPS)   │     │  Frame Grab  │     │  Emotion Analysis  │     │  on Live Frame  │
│             │     │              │     │                    │     │                 │
└─────────────┘     └──────────────┘     └────────────────────┘     └─────────────────┘
                                                   │
                                     ┌─────────────▼──────────────┐
                                     │   VGG-Face Neural Network   │
                                     │   (runs locally, offline)   │
                                     │                             │
                                     │  Input: 48x48 face crop     │
                                     │  Output: 7 emotion scores   │
                                     └─────────────────────────────┘
```

### The pipeline in detail:

1. **Frame Capture** — OpenCV grabs a frame from your webcam at up to 30 FPS
2. **Face Detection** — DeepFace locates faces using a built-in face detector (MTCNN / RetinaFace)
3. **Emotion Inference** — Each detected face is cropped, resized to 48×48, and passed through the VGG-Face emotion classifier
4. **Score Output** — Returns confidence percentages for all 7 emotions (sums to ~100%)
5. **Smoothing** — The last 5 dominant emotion predictions are stored; the most frequent one is displayed to avoid jitter
6. **Rendering** — Bounding boxes, labels, confidence bars, and the HUD are composited onto the frame using OpenCV drawing functions
7. **Display** — The final annotated frame is shown in a live window at full resolution

---

## 📁 Project Structure

```
realtime-emotion-detector/
│
├── app.py                  # 🧠 Main application — all logic lives here
├── requirements.txt        # 📦 Python dependencies
├── .gitignore              # 🚫 Files excluded from git
├── LICENSE                 # 📄 MIT License
├── README.md               # 📖 You are here
│
└── screenshots/            # 📸 Auto-created when you press S
    └── emotion_20250101_120000.jpg
```

---

## 🛠️ Tech Stack

| Technology | Version | Purpose |
|---|---|---|
| [Python](https://www.python.org/) | 3.9+ | Core language |
| [DeepFace](https://github.com/serengil/deepface) | 0.0.93 | Pre-trained emotion recognition (VGG-Face) |
| [OpenCV](https://opencv.org/) | 4.9 | Webcam capture, frame processing, drawing |
| [NumPy](https://numpy.org/) | 1.26 | Array operations |
| [TF-Keras](https://keras.io/) | latest | Backend for DeepFace model inference |

---

## 🤔 Frequently Asked Questions

**Q: Does this send my video to the internet?**
> No. Everything runs locally on your machine. No data is ever transmitted.

**Q: Why is it slow on first run?**
> DeepFace downloads the model weights (~100MB) automatically. After that, it's fast.

**Q: My webcam isn't detected. What do I do?**
> Make sure no other app is using your webcam. Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in `app.py` if you have multiple cameras.

**Q: Can I use this without a GPU?**
> Yes! It runs on CPU only. With a GPU (CUDA), it will be significantly faster.

**Q: How accurate is it?**
> DeepFace's emotion model achieves ~82% accuracy on the FER-2013 benchmark dataset. Real-world accuracy depends on lighting and camera quality.

---

## 🗺️ Roadmap

- [ ] 📈 Emotion timeline graph — track mood changes over time
- [ ] 💾 Export emotions to CSV for data analysis
- [ ] 🎵 Mood-based Spotify playlist trigger
- [ ] 🌐 Streamlit web interface (no OpenCV window needed)
- [ ] 👥 Multi-face tracking with unique IDs per person
- [ ] 🪞 Raspberry Pi deployment for a smart mirror build
- [ ] 📱 Mobile version via webcam stream

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'feat: add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
Free to use, modify, and distribute.

---

## 🙏 Acknowledgements

- [DeepFace](https://github.com/serengil/deepface) by Sefik Ilkin Serengil — the incredible face analysis library that makes this possible
- [FER-2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013) — the dataset used to train the emotion model
- [OpenCV](https://opencv.org/) — the backbone of all computer vision work

---

<div align="center">

**⭐ If you found this useful, please star the repo — it helps a lot!**

Made with ❤️ and Python

![Footer](https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer)

</div>
