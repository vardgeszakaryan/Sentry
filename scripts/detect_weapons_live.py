"""
detect_weapons_live.py
----------------------
Real-time weapon detection — Sentry model (best.pt).

Architecture (from niga2.py)
----------------------------
- Dedicated capture thread  → LatestFrame shared buffer (always-fresh, never blocks)
- Dedicated inference thread → weapon model with hold/hysteresis stabiliser
- Main display loop          → cinematic post-process + FBI HUD, runs at full speed

Stabiliser logic
----------------
- ENTER_CONF: must exceed this to "lock on" to a detection
- EXIT_CONF:  model still runs at lower threshold; below ENTER_CONF increments miss count
- HOLD_FRAMES: detection held for this many missed frames before clearing

Cinematic FX
------------
- 1280×1280 display, teal-orange LUT, vignette, scan lines, film grain
- Glowing corner-bracket weapon markers with pulsing glow + crosshair
- FBI HUD: timestamp, REC blinker, threat counter, HOLD indicator

Usage
-----
    python scripts/detect_weapons_live.py
    python scripts/detect_weapons_live.py --weights releases/v1.0.0/best.pt --camera 1
"""

import argparse
import math
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ── Resolution ────────────────────────────────────────────────────────────────
DISPLAY_SIZE  = 1280
CAM_W, CAM_H  = 1280, 720       # native capture resolution

# ── Stabiliser config ─────────────────────────────────────────────────────────
ENTER_CONF   = 0.35
EXIT_CONF    = 0.20
HOLD_FRAMES  = 12
TRACK_IOU    = 0.70              # niga2.py uses 0.7 — more stable tracking

# ── Palette (BGR) ─────────────────────────────────────────────────────────────
RED_NEON  = (0,  50, 220)
AMBER     = (0, 180, 255)
TEAL      = (180, 200, 80)
GREY      = (150, 150, 150)
FONT      = cv2.FONT_HERSHEY_SIMPLEX


# ── EMA Smoother ──────────────────────────────────────────────────────────────
class EMASmoother:
    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
        self._state: dict[int, list[float]] = {}

    def smooth(self, tid: int, vals: list[float]) -> list[float]:
        if tid not in self._state:
            self._state[tid] = vals[:]
        else:
            p = self._state[tid]
            self._state[tid] = [self.alpha*v + (1-self.alpha)*q
                                 for v, q in zip(vals, p)]
        return self._state[tid]

    def evict(self, active: set[int]) -> None:
        for k in [k for k in self._state if k not in active]:
            del self._state[k]


# ── LatestFrame buffer (from niga2.py) ───────────────────────────────────────
class LatestFrame:
    def __init__(self):
        self._lock  = threading.Lock()
        self._frame = None
        self._stop  = False

    def update(self, frame):
        with self._lock:
            self._frame = frame

    def read(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self):
        with self._lock:
            self._stop = True

    def stopped(self):
        with self._lock:
            return self._stop


def _put_latest(q: queue.Queue, item) -> None:
    """1-slot queue — discard stale result, insert newest."""
    try:
        if q.full():
            q.get_nowait()
    except queue.Empty:
        pass
    try:
        q.put_nowait(item)
    except queue.Full:
        pass


# ── Detection result dataclass ────────────────────────────────────────────────
@dataclass
class WeaponResult:
    dets:  list = field(default_factory=list)  # [(smooth_box, tid, label, conf)]
    held:  bool = False
    miss:  int  = 0


# ── Capture thread ────────────────────────────────────────────────────────────
def capture_loop(cap: cv2.VideoCapture, buf: LatestFrame) -> None:
    while not buf.stopped():
        ok, frame = cap.read()
        if not ok:
            buf.stop()
            break
        buf.update(frame)
        time.sleep(0.001)


# ── Weapon inference thread ───────────────────────────────────────────────────
def weapon_worker(buf: LatestFrame, out_q: queue.Queue,
                  model: YOLO, ema: float) -> None:
    smoother   = EMASmoother(alpha=ema)
    last_good  = None    # last accepted WeaponResult
    missed     = 0

    while not buf.stopped():
        frame = buf.read()
        if frame is None:
            time.sleep(0.005)
            continue

        # Resize to square for model
        sq = cv2.resize(frame, (DISPLAY_SIZE, DISPLAY_SIZE))

        results = model.track(
            source=sq,
            conf=EXIT_CONF,
            iou=TRACK_IOU,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )
        r0 = results[0]

        dets       = []
        active_ids = set()
        accepted   = False

        if r0.boxes is not None and len(r0.boxes) > 0:
            best_conf = float(r0.boxes.conf.max())
            if best_conf >= ENTER_CONF:
                accepted = True
                missed   = 0
                for i, box in enumerate(r0.boxes.xyxy):
                    c   = float(r0.boxes.conf[i])
                    lbl = model.names[int(r0.boxes.cls[i])]
                    tid = int(r0.boxes.id[i]) if r0.boxes.id is not None else -1
                    raw = box.tolist()
                    sb  = smoother.smooth(tid, raw) if tid >= 0 else raw
                    if tid >= 0:
                        active_ids.add(tid)
                    dets.append((sb, tid, lbl, c))
            else:
                missed += 1
        else:
            missed += 1

        smoother.evict(active_ids)

        if accepted:
            res = WeaponResult(dets=dets, held=False, miss=0)
            last_good = res
        elif last_good is not None and missed <= HOLD_FRAMES:
            res = WeaponResult(dets=last_good.dets, held=True, miss=missed)
        else:
            if missed > HOLD_FRAMES:
                last_good = None
            res = WeaponResult(dets=[], held=False, miss=missed)

        _put_latest(out_q, res)


# ── Cinematic post-processing ─────────────────────────────────────────────────
def _make_vignette(size: int) -> np.ndarray:
    cx = cy = size // 2
    Y, X = np.ogrid[:size, :size]
    dist = np.sqrt((X-cx)**2 + (Y-cy)**2) / (size * 0.7)
    return (1.0 - np.clip(dist**1.8, 0, 1)).astype(np.float32)

def _make_scanlines(size: int) -> np.ndarray:
    m = np.ones((size, size), dtype=np.float32)
    m[::2, :] = 0.83
    return m

def _build_luts():
    x = np.arange(256, dtype=np.float32)
    return (np.clip(x*0.80+22, 0, 255).astype(np.uint8),
            np.clip(x*0.88+8,  0, 255).astype(np.uint8),
            np.clip((x/255)**0.85*245, 0, 255).astype(np.uint8))

_VIG  = _make_vignette(DISPLAY_SIZE)
_SL   = _make_scanlines(DISPLAY_SIZE)
_BLUT, _GLUT, _RLUT = _build_luts()

def apply_cinematic(frame: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(frame)
    frame = cv2.merge([cv2.LUT(b, _BLUT), cv2.LUT(g, _GLUT), cv2.LUT(r, _RLUT)])
    f32 = frame.astype(np.float32)
    f32 *= _VIG[:, :, np.newaxis]
    f32 *= _SL[:, :, np.newaxis]
    f32 = np.clip(f32 + np.random.normal(0, 5, f32.shape), 0, 255)
    return f32.astype(np.uint8)


# ── Drawing ───────────────────────────────────────────────────────────────────
def _glow_line(img, p1, p2, color, thickness=2) -> None:
    ov = np.zeros_like(img)
    cv2.line(ov, p1, p2, color, thickness+6, cv2.LINE_AA)
    cv2.addWeighted(img, 1.0, cv2.GaussianBlur(ov, (11,11), 0), 0.7, 0, img)
    cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)

def draw_bracket_box(frame, box, tid, label, conf, t, held=False) -> None:
    x1, y1, x2, y2 = (int(v) for v in box)
    w, h = x2-x1, y2-y1
    arm  = max(14, min(w,h)//4)

    pulse = 0.55 + 0.45*math.sin(t * (4 if held else 7))
    col   = tuple(int(c*pulse) for c in (RED_NEON if not held else (0,180,180)))
    amb   = tuple(int(c*pulse) for c in AMBER)

    for (cx,cy),(dx,dy) in [((x1,y1),(1,1)),((x2,y1),(-1,1)),
                              ((x1,y2),(1,-1)),((x2,y2),(-1,-1))]:
        _glow_line(frame,(cx,cy),(cx+dx*arm,cy),col)
        _glow_line(frame,(cx,cy),(cx,cy+dy*arm),col)

    mx,my = (x1+x2)//2,(y1+y2)//2
    for off in range(-18,19,12):
        cv2.line(frame,(mx+off,my),(mx+off+8,my),amb,1,cv2.LINE_AA)
        cv2.line(frame,(mx,my+off),(mx,my+off+8),amb,1,cv2.LINE_AA)

    tag = (f"[HOLD] #{tid:02d} {label.upper()} {conf:.0%}"
           if held else f"#{tid:02d} {label.upper()} {conf:.0%}")
    (tw,th),bl = cv2.getTextSize(tag, FONT, 0.52, 1)
    cv2.rectangle(frame,(x1,y1-th-bl-6),(x1+tw+4,y1),(0,0,0),-1)
    cv2.putText(frame, tag,(x1+2,y1-bl-2), FONT,0.52,AMBER,1,cv2.LINE_AA)

def draw_hud(frame, fps, det_count, held, t) -> None:
    h, w = frame.shape[:2]
    arm = 42
    for (cx,cy),(dx,dy) in [((8,8),(1,1)),((w-8,8),(-1,1)),
                              ((8,h-8),(1,-1)),((w-8,h-8),(-1,-1))]:
        cv2.line(frame,(cx,cy),(cx+dx*arm,cy),TEAL,2,cv2.LINE_AA)
        cv2.line(frame,(cx,cy),(cx,cy+dy*arm),TEAL,2,cv2.LINE_AA)

    cv2.putText(frame, time.strftime("  %Y-%m-%d  %H:%M:%S"),
                (54,28), FONT,0.55,TEAL,1,cv2.LINE_AA)

    if int(t*2)%2==0:
        cv2.circle(frame,(w-60,20),6,(0,0,200),-1,cv2.LINE_AA)
        cv2.putText(frame,"REC",(w-48,25),FONT,0.5,(0,0,200),1,cv2.LINE_AA)

    cv2.putText(frame,f"FPS {fps:5.1f}",(54,h-18),FONT,0.52,GREY,1,cv2.LINE_AA)

    tc    = AMBER if det_count==0 else RED_NEON
    tlbl  = f"THREATS: {det_count:02d}" + ("  [HOLD]" if held else "")
    (tw,_),_ = cv2.getTextSize(tlbl,FONT,0.58,1)
    cv2.putText(frame,tlbl,(w-tw-54,h-18),FONT,0.58,tc,1,cv2.LINE_AA)

    sl = "[ SENTRY AI  //  WEAPON DETECTION  //  v1.0.0 ]"
    (tw,_),_ = cv2.getTextSize(sl,FONT,0.45,1)
    cv2.putText(frame,sl,((w-tw)//2,22),FONT,0.45,GREY,1,cv2.LINE_AA)


# ── Main ──────────────────────────────────────────────────────────────────────
def run(weights: str, camera: int, ema: float) -> None:
    p = Path(weights)
    if not p.exists():
        print(f"[ERROR] Weights not found: {p}"); sys.exit(1)

    print(f"[INFO] Loading model: {p}")
    model = YOLO(str(p))

    print(f"[INFO] Opening camera {camera}")
    cap = cv2.VideoCapture(camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {camera}"); sys.exit(1)

    buf      = LatestFrame()
    weapon_q = queue.Queue(maxsize=1)

    threading.Thread(target=capture_loop,  args=(cap, buf),               daemon=True).start()
    threading.Thread(target=weapon_worker, args=(buf, weapon_q, model, ema), daemon=True).start()

    latest_res = WeaponResult()
    prev_time  = time.time()
    fps        = 0.0
    t0         = time.time()

    print("[INFO] Running — press Q to quit")
    while True:
        try:
            latest_res = weapon_q.get_nowait()
        except queue.Empty:
            pass

        frame = buf.read()
        if frame is None:
            time.sleep(0.005)
            continue

        display = cv2.resize(frame, (DISPLAY_SIZE, DISPLAY_SIZE))
        t       = time.time() - t0

        for (box, tid, lbl, conf) in latest_res.dets:
            draw_bracket_box(display, box, tid, lbl, conf, t, latest_res.held)

        display = apply_cinematic(display)
        draw_hud(display, fps, len(latest_res.dets), latest_res.held, t)

        now = time.time()
        fps = 0.9*fps + 0.1/max(now-prev_time, 1e-6)
        prev_time = now

        cv2.imshow("SENTRY // WEAPON DETECTION", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if buf.stopped():
            break

    buf.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Stopped cleanly.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="releases/v1.0.0/best.pt")
    p.add_argument("--camera",  type=int,   default=0)
    p.add_argument("--ema",     type=float, default=0.4)
    return p.parse_args()

if __name__ == "__main__":
    a = parse_args()
    run(a.weights, a.camera, a.ema)
