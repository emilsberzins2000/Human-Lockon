import cv2
import numpy as np
import time
import math
import threading
from ultralytics import YOLO

# ==========================================================
# GTA-style lock-on camera (FAST / INSTANT LOCK FEEL)
# + FULLSCREEN TOGGLE ADDED (your "old code" + fullscreen)
#
# Controls:
#   F = toggle fullscreen
#   Esc = if fullscreen -> exit fullscreen, else quit
#   L = toggle lock-on
#   C = clear target
#   + / - = manual zoom multiplier
#   B = toggle sound
#   Q = quit
# ==========================================================

# ---------------------------
# VIDEO / VIEW
# ---------------------------
CAM_W, CAM_H = 640, 480
VIEW_W, VIEW_H = 640, 480

WINDOW_NAME = "GTA Lock-On (Instant Feel)"
FULLSCREEN_START = False

# ---------------------------
# DETECTION
# ---------------------------
CONF_THRES = 0.35
MAX_DET = 12

# ---------------------------
# AIM POINT (CHEST)
# ---------------------------
CHEST_Y = 0.40

# ---------------------------
# CAMERA FOLLOW (FAST)
# ---------------------------
DEADZONE = 0.00     # remove deadzone so it starts moving immediately
SPRING = 55.0       # MUCH higher = snaps to target fast
DAMPING = 0.60      # lower damping = faster response (less floaty)

# Optional: hard snap limit per frame (prevents overshoot/jitter)
MAX_CAM_STEP_PX = 80  # cap how many pixels camera center can move per frame

# ---------------------------
# LOCK LOGIC (FASTER)
# ---------------------------
LOCK_ON = True
STICKY_LOCK = True

LOCK_RADIUS_PX = 150      # big circle so it counts as "on target" sooner
LOCK_HOLD_FRAMES = 3      # basically instant (was ~14)
JUMP_RESET_PX = 120       # tolerate more movement without resetting

# ---------------------------
# ADAPTIVE ZOOM (still)
# ---------------------------
ZOOM_IDLE = 1.15
ZOOM_CLOSE = 1.08
ZOOM_FAR = 2.15
ZOOM_SPEED = 9.0          # zoom changes faster
ZOOM_MANUAL_MULT = 1.00

AREA_CLOSE = 0.30
AREA_FAR = 0.03

# ---------------------------
# SOUND (SYNTH, NO MP3)
# ---------------------------
SOUND_ENABLE = True
FS = 44100

FREQ_LOW = 600
FREQ_HIGH = 1000  # static when locked

BEEP_INTERVAL_SLOW = 0.28  # faster start
BEEP_INTERVAL_FAST = 0.06  # very fast when locked
BEEP_DUR_MS = 55

DOUBLE_PULSE = True
SECOND_PULSE_GAP_MS = 28
SECOND_PULSE_DUR_MS = 35
SECOND_PULSE_LEVEL = 0.55

CHIRP_MULT = 1.08


# ==========================================================
# Fullscreen helper
# ==========================================================
def set_fullscreen(window_name: str, fullscreen: bool):
    """
    OpenCV fullscreen toggle (works depending on OS/backend).
    """
    try:
        cv2.setWindowProperty(
            window_name,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
        )
    except Exception:
        pass


# ==========================================================
# Helpers
# ==========================================================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def crop_with_center(frame, cx, cy, crop_w, crop_h):
    h, w = frame.shape[:2]
    x1 = int(cx - crop_w / 2)
    y1 = int(cy - crop_h / 2)
    x1 = clamp(x1, 0, w - crop_w)
    y1 = clamp(y1, 0, h - crop_h)
    x2 = x1 + crop_w
    y2 = y1 + crop_h
    return frame[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)

def chest_point(x1, y1, x2, y2):
    cx = (x1 + x2) / 2.0
    cy = y1 + CHEST_Y * (y2 - y1)
    return cx, cy

def project_point_to_view(px, py, crop_x1, crop_y1, crop_w, crop_h, vw, vh):
    nx = (px - crop_x1) / float(crop_w)
    ny = (py - crop_y1) / float(crop_h)
    vx = int(nx * vw)
    vy = int(ny * vh)
    return vx, vy, nx, ny

def draw_x_reticle(img, center, size=14, thickness=2):
    x, y = center
    cv2.line(img, (x - size, y - size), (x + size, y + size), (255, 255, 255), thickness)
    cv2.line(img, (x - size, y + size), (x + size, y - size), (255, 255, 255), thickness)

def map_zoom_from_area(area_ratio):
    ar = float(np.clip(area_ratio, AREA_FAR, AREA_CLOSE))
    t = (ar - AREA_FAR) / max(1e-6, (AREA_CLOSE - AREA_FAR))  # 0 far -> 1 close
    return ZOOM_FAR + (ZOOM_CLOSE - ZOOM_FAR) * t

def pick_target(dets, prev_center=None):
    if not dets:
        return None
    centers = []
    areas = []
    for (x1, y1, x2, y2) in dets:
        centers.append(chest_point(x1, y1, x2, y2))
        areas.append((x2 - x1) * (y2 - y1))
    if prev_center is not None:
        px, py = prev_center
        d2 = [((cx - px) ** 2 + (cy - py) ** 2) for (cx, cy) in centers]
        idx = int(np.argmin(d2))
    else:
        idx = int(np.argmax(areas))
    x1, y1, x2, y2 = dets[idx]
    tcx, tcy = centers[idx]
    return (tcx, tcy, x1, y1, x2, y2, float(areas[idx]))


# ==========================================================
# Sound Engine
# ==========================================================
class LockSound(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._lock = threading.Lock()
        self._stop = False
        self.enabled = True
        self.active = False
        self.progress = 0.0
        self.locked = False

        self.backend = None
        self._winsound = None
        self._sa = None

        try:
            import winsound
            self._winsound = winsound
            self.backend = "winsound"
        except Exception:
            self.backend = None

        if self.backend is None:
            try:
                import simpleaudio as sa
                self._sa = sa
                self.backend = "simpleaudio"
            except Exception:
                self.backend = None

        self._last_beep = 0.0
        self._cache = {}

    def set_state(self, enabled: bool, active: bool, progress: float, locked: bool):
        with self._lock:
            self.enabled = bool(enabled)
            self.active = bool(active)
            self.progress = float(np.clip(progress, 0.0, 1.0))
            self.locked = bool(locked)

    def stop(self):
        self._stop = True

    def _interval(self, p, locked):
        if locked:
            return BEEP_INTERVAL_FAST
        return BEEP_INTERVAL_SLOW - (BEEP_INTERVAL_SLOW - BEEP_INTERVAL_FAST) * p

    def _freq(self, p, locked):
        if locked:
            return int(FREQ_HIGH)  # static 1000Hz when locked
        return int(FREQ_LOW + (FREQ_HIGH - FREQ_LOW) * p)

    def _wave_chirp(self, f0, f1, dur_ms, level=1.0):
        key = (int(f0), int(f1), int(dur_ms), int(level * 1000))
        if key in self._cache:
            return self._cache[key]
        dur_s = dur_ms / 1000.0
        n = max(1, int(FS * dur_s))
        t = np.linspace(0, dur_s, n, endpoint=False)
        k = (f1 - f0) / max(1e-6, dur_s)
        phase = 2 * np.pi * (f0 * t + 0.5 * k * t * t)
        wave = np.sin(phase).astype(np.float32)

        attack = max(1, int(0.12 * n))
        decay = max(1, int(0.25 * n))
        env = np.ones(n, dtype=np.float32)
        env[:attack] = np.linspace(0, 1, attack)
        env[-decay:] = np.linspace(1, 0, decay)

        wave = wave * env * (0.22 * level)
        audio = np.int16(np.clip(wave, -1, 1) * 32767)
        self._cache[key] = audio
        return audio

    def _play_simpleaudio(self, freq):
        f0 = freq
        f1 = int(freq * CHIRP_MULT)
        main = self._wave_chirp(f0, f1, BEEP_DUR_MS, level=1.0)
        if DOUBLE_PULSE:
            gap = np.zeros(int(FS * (SECOND_PULSE_GAP_MS / 1000.0)), dtype=np.int16)
            second = self._wave_chirp(int(freq * 0.92), int(freq * 0.98), SECOND_PULSE_DUR_MS, level=SECOND_PULSE_LEVEL)
            audio = np.concatenate([main, gap, second])
        else:
            audio = main
        self._sa.play_buffer(audio, 1, 2, FS)

    def _play_winsound(self, freq):
        try:
            self._winsound.Beep(int(freq), int(BEEP_DUR_MS))
            if DOUBLE_PULSE:
                time.sleep(SECOND_PULSE_GAP_MS / 1000.0)
                self._winsound.Beep(int(freq * 0.92), int(SECOND_PULSE_DUR_MS))
        except Exception:
            pass

    def run(self):
        while not self._stop:
            time.sleep(0.004)
            with self._lock:
                enabled = self.enabled and SOUND_ENABLE
                active = self.active
                p = self.progress
                locked = self.locked
                backend = self.backend
            if not enabled or not active:
                continue
            interval = self._interval(p, locked)
            freq = self._freq(p, locked)
            now = time.time()
            if (now - self._last_beep) < interval:
                continue
            self._last_beep = now
            if backend == "winsound":
                self._play_winsound(freq)
            elif backend == "simpleaudio":
                self._play_simpleaudio(freq)


# ==========================================================
# Main
# ==========================================================
def main():
    global LOCK_ON, ZOOM_MANUAL_MULT, SOUND_ENABLE

    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    cam_cx, cam_cy = CAM_W // 2, CAM_H // 2
    vel_x, vel_y = 0.0, 0.0

    locked_center = None
    locked_box = None

    lock_counter = 0
    fully_locked = False

    zoom = ZOOM_IDLE
    prev_t = time.time()

    sound = LockSound()
    sound.start()

    # ---- FULLSCREEN SETUP (ADDED) ----
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, VIEW_W, VIEW_H)
    fullscreen = FULLSCREEN_START
    if fullscreen:
        set_fullscreen(WINDOW_NAME, True)

    print("FAST LOCK MODE")
    print("Controls: F fullscreen | L toggle | C clear | +/- zoom mult | B sound | Q/Esc quit")
    if sound.backend is None and SOUND_ENABLE:
        print("No sound backend found. On non-Windows: pip install simpleaudio")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        now = time.time()
        dt = max(1e-3, now - prev_t)
        prev_t = now

        # Detect persons
        res = model.predict(
            source=frame,
            conf=CONF_THRES,
            classes=[0],
            max_det=MAX_DET,
            verbose=False
        )[0]

        dets = []
        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy()
            for b in boxes:
                x1, y1, x2, y2 = b
                dets.append((float(x1), float(y1), float(x2), float(y2)))

        # Choose/keep target
        prev_for_pick = locked_center if (LOCK_ON and STICKY_LOCK and locked_center is not None) else None
        target = pick_target(dets, prev_center=prev_for_pick)

        target_changed = False
        area_ratio = None

        if LOCK_ON and target is not None:
            tcx, tcy, x1, y1, x2, y2, area = target
            if locked_center is None:
                target_changed = True
            else:
                if math.hypot(tcx - locked_center[0], tcy - locked_center[1]) > JUMP_RESET_PX:
                    target_changed = True
            locked_center = (tcx, tcy)
            locked_box = (x1, y1, x2, y2)
            area_ratio = area / float(CAM_W * CAM_H)
        else:
            target_changed = True
            if not LOCK_ON:
                locked_center = None
                locked_box = None

        if target_changed:
            lock_counter = 0
            fully_locked = False

        # Fast adaptive zoom
        if LOCK_ON and area_ratio is not None:
            want_zoom = map_zoom_from_area(area_ratio)
        else:
            want_zoom = ZOOM_IDLE

        want_zoom *= ZOOM_MANUAL_MULT
        want_zoom = float(np.clip(want_zoom, 1.0, 2.6))
        zoom += (want_zoom - zoom) * clamp(ZOOM_SPEED * dt, 0.0, 1.0)

        crop_w = clamp(int(CAM_W / zoom), 140, CAM_W)
        crop_h = clamp(int(CAM_H / zoom), 110, CAM_H)

        # Desired camera center
        if LOCK_ON and locked_center is not None:
            desired_cx, desired_cy = int(locked_center[0]), int(locked_center[1])
        else:
            desired_cx, desired_cy = cam_cx, cam_cy

        # Fast follow (spring + clamp max step)
        err_x = desired_cx - cam_cx
        err_y = desired_cy - cam_cy

        # No deadzone (instant move)
        acc_x = SPRING * err_x
        acc_y = SPRING * err_y

        vel_x = (vel_x + acc_x * dt) * DAMPING
        vel_y = (vel_y + acc_y * dt) * DAMPING

        step_x = int(vel_x * dt)
        step_y = int(vel_y * dt)

        # Clamp movement per frame -> near-instant but stable
        step_x = clamp(step_x, -MAX_CAM_STEP_PX, MAX_CAM_STEP_PX)
        step_y = clamp(step_y, -MAX_CAM_STEP_PX, MAX_CAM_STEP_PX)

        cam_cx += step_x
        cam_cy += step_y

        cam_cx = clamp(cam_cx, crop_w // 2, CAM_W - crop_w // 2)
        cam_cy = clamp(cam_cy, crop_h // 2, CAM_H - crop_h // 2)

        # Render view
        view, (x1c, y1c, x2c, y2c) = crop_with_center(frame, cam_cx, cam_cy, crop_w, crop_h)
        view = cv2.resize(view, (VIEW_W, VIEW_H), interpolation=cv2.INTER_LINEAR)

        vh, vw = view.shape[:2]
        ret_center = (vw // 2, vh // 2)
        draw_x_reticle(view, ret_center, size=14, thickness=2)

        # Lock progress
        progress = 0.0
        sound_active = False

        if LOCK_ON and locked_center is not None:
            sound_active = True
            tcx, tcy = locked_center
            vx, vy, nx, ny = project_point_to_view(tcx, tcy, x1c, y1c, crop_w, crop_h, vw, vh)

            if 0 <= nx <= 1 and 0 <= ny <= 1:
                d = math.hypot(vx - ret_center[0], vy - ret_center[1])

                if d <= LOCK_RADIUS_PX:
                    lock_counter = min(LOCK_HOLD_FRAMES, lock_counter + 1)
                else:
                    lock_counter = max(0, lock_counter - 2)

                fully_locked = (lock_counter >= LOCK_HOLD_FRAMES)
                progress = lock_counter / float(max(1, LOCK_HOLD_FRAMES))
            else:
                lock_counter = 0
                fully_locked = False
                progress = 0.0
        else:
            lock_counter = 0
            fully_locked = False
            progress = 0.0
            sound_active = False

        sound.set_state(SOUND_ENABLE, sound_active, progress, fully_locked)

        # Draw detections (thin)
        for (bx1, by1, bx2, by2) in dets:
            nx1 = (bx1 - x1c) / float(crop_w)
            ny1 = (by1 - y1c) / float(crop_h)
            nx2 = (bx2 - x1c) / float(crop_w)
            ny2 = (by2 - y1c) / float(crop_h)
            if nx2 < 0 or ny2 < 0 or nx1 > 1 or ny1 > 1:
                continue
            xA = clamp(int(nx1 * vw), 0, vw - 1)
            yA = clamp(int(ny1 * vh), 0, vh - 1)
            xB = clamp(int(nx2 * vw), 0, vw - 1)
            yB = clamp(int(ny2 * vh), 0, vh - 1)
            cv2.rectangle(view, (xA, yA), (xB, yB), (0, 200, 255), 1)

        # Draw locked target
        if LOCK_ON and locked_box is not None and locked_center is not None:
            bx1, by1, bx2, by2 = locked_box
            nx1 = (bx1 - x1c) / float(crop_w)
            ny1 = (by1 - y1c) / float(crop_h)
            nx2 = (bx2 - x1c) / float(crop_w)
            ny2 = (by2 - y1c) / float(crop_h)

            xA = clamp(int(nx1 * vw), 0, vw - 1)
            yA = clamp(int(ny1 * vh), 0, vh - 1)
            xB = clamp(int(nx2 * vw), 0, vw - 1)
            yB = clamp(int(ny2 * vh), 0, vh - 1)

            if fully_locked:
                box_color = (0, 0, 255)
                label = "LOCKED"
            else:
                box_color = (0, 255, 255)
                label = f"LOCKING {int(progress * 100)}%"

            cv2.rectangle(view, (xA, yA), (xB, yB), box_color, 2)

            tcx, tcy = locked_center
            vx, vy, nx, ny = project_point_to_view(tcx, tcy, x1c, y1c, crop_w, crop_h, vw, vh)
            if 0 <= nx <= 1 and 0 <= ny <= 1:
                cv2.circle(view, (vx, vy), 6 if fully_locked else 5, box_color, -1)

            cv2.putText(view, label, (xA, max(22, yA - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

            cv2.circle(view, ret_center, LOCK_RADIUS_PX, box_color, 2)

        # HUD
        cv2.putText(view, "FAST LOCK", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(view, f"Zoom: {zoom:.2f}  Manual x{ZOOM_MANUAL_MULT:.2f}", (12, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(view, f"Sound: {'ON' if SOUND_ENABLE else 'OFF'}  Backend: {sound.backend or 'none'}",
                    (12, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(view, "F fullscreen | ESC exit fs | Q quit", (12, vh - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # ---- show using fullscreen-capable window (CHANGED) ----
        cv2.imshow(WINDOW_NAME, view)

        k = cv2.waitKey(1) & 0xFF

        # ---- FULLSCREEN KEYS (ADDED) ----
        if k in (ord('f'), ord('F')):
            fullscreen = not fullscreen
            set_fullscreen(WINDOW_NAME, fullscreen)

        # ESC: exit fullscreen first, otherwise quit
        if k == 27:
            if fullscreen:
                fullscreen = False
                set_fullscreen(WINDOW_NAME, fullscreen)
                continue
            else:
                break

        # Existing keys
        if k in (ord('q'), ord('Q')):
            break
        if k in (ord('l'), ord('L')):
            LOCK_ON = not LOCK_ON
            locked_center = None
            locked_box = None
            lock_counter = 0
            fully_locked = False
        if k in (ord('c'), ord('C')):
            locked_center = None
            locked_box = None
            lock_counter = 0
            fully_locked = False
        if k in (ord('+'), ord('=')):  # = is usually same key as +
            ZOOM_MANUAL_MULT = float(np.clip(ZOOM_MANUAL_MULT + 0.05, 0.75, 1.75))
        if k in (ord('-'), ord('_')):
            ZOOM_MANUAL_MULT = float(np.clip(ZOOM_MANUAL_MULT - 0.05, 0.75, 1.75))
        if k in (ord('b'), ord('B')):
            SOUND_ENABLE = not SOUND_ENABLE

    cap.release()
    sound.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
