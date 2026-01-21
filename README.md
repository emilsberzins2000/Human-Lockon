


````markdown
# Human-LockOn (Real-Time Human Tracking)

🎯 **Real-time AI camera system with GTA-style lock-on mechanics**

This project implements a **fast, responsive lock-on camera** inspired by GTA missile targeting systems.  
It detects humans in real time, locks onto the chest area, zooms dynamically, snaps the camera instantly, and plays a synthesized lock-on sound — all without using any external audio files.

---

## ✨ Features

- ✅ **Real-time human detection** (YOLOv8)
- 🎯 **Chest-based targeting** (more realistic aiming)
- ⚡ **Instant snap camera movement** (very fast lock)
- 🔴 **Lock indicator**
  - Yellow = locking
  - Red = fully locked
- 🔊 **Synthesized lock-on sound**
  - Frequency ramps from **600 Hz → 1000 Hz**
  - Stays at **static 1000 Hz** when fully locked
- 🔍 **Adaptive zoom**
  - Target far → zoom in
  - Target close → zoom out
- 🖥 **Fullscreen support**
  - Press **F** to toggle fullscreen
  - Press **ESC** to exit fullscreen
- 🎮 **GTA-like feel**, built fully in Python
---
## 🎮 Controls

| Key | Action |
|----|-------|
| `F` | Toggle fullscreen |
| `ESC` | Exit fullscreen (or quit if not fullscreen) |
| `L` | Toggle lock-on |
| `C` | Clear target |
| `+` | Increase zoom multiplier |
| `-` | Decrease zoom multiplier |
| `B` | Toggle sound |
| `Q` | Quit |

---

## 🛠 Requirements

- Python **3.9+**
- Webcam
- OS: Windows / Linux / macOS

### Python dependencies
```bash
pip install ultralytics opencv-python numpy
````

Optional (for sound on non-Windows systems):

```bash
pip install simpleaudio
```

---

## 🚀 How to Run

```bash
python main.py
```

Make sure your webcam is connected and accessible.

---

## 💰 License & Purchase

Read LICENSE-DEMO

---

### 💵 Price

**10 EUR** — one-time purchase
Includes:

* Full source code
* Commercial usage rights
* Future bug-fix updates (minor)

---

## 📩 Contact / Purchase

For purchase, licensing questions, or demos:

📷 **Instagram:** `@randomguygithub`

DM me directly.

---

## ⚠ Disclaimer

This project is for **educational, research, and commercial use**.
The author is not responsible for misuse or illegal deployment.

---

## ⭐ Final Notes

If you want:

* Controller support
* Target switching
* Fire / trigger logic
* Multiple target cycling
* Aim assist modes

👉 Contact me on Instagram.

