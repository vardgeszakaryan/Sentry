"""
detect_combined_live.py
-----------------------
Real-time combined detection:
  • YOLOv26 pose model  → glowing neon COCO-17 skeleton
  • Sentry best.pt      → FBI corner-bracket weapon markers

Architecture (from niga2.py)
----------------------------
- Dedicated capture thread  → LatestFrame shared buffer (always-fresh, never blocks)
- Dedicated weapon thread   → inference + ENTER/EXIT hysteresis + HOLD stabiliser
- Dedicated pose thread     → inference, outputs latest keypoints
- Main display loop         → reads latest results, draws overlays, applies cinematic FX

Cinematic FX
------------
- 1280×1280, teal-orange LUT, vignette, scan lines, film grain
- Glowing neon skeleton with EMA-smoothed keypoints
- Pulsing corner-bracket weapon markers, crosshair, HOLD indicator
- FBI HUD: timestamp, REC blinker, body/threat counters

Usage
-----
    python scripts/detect_combined_live.py
    python scripts/detect_combined_live.py ^
        --weapon-weights releases/v1.0.0/best.pt ^
        --pose-weights   releases/v1.0.0/yolo26n-pose.pt ^
        --camera 1
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
import torch
from ultralytics import YOLO

# ── Resolution ────────────────────────────────────────────────────────────────
DISPLAY_SIZE = 1280
CAM_W, CAM_H = 1280, 720

# ── Stabiliser config ─────────────────────────────────────────────────────────
ENTER_CONF  = 0.35
EXIT_CONF   = 0.20
HOLD_FRAMES = 12
TRACK_IOU   = 0.70

# ── COCO-17 skeleton ──────────────────────────────────────────────────────────
SKELETON = [
    (0,1),(0,2),(1,3),(2,4),
    (3,5),(4,6),
    (5,6),(5,11),(6,12),(11,12),
    (5,7),(7,9),(6,8),(8,10),
    (11,13),(13,15),(12,14),(14,16),
]

# ── Palette ───────────────────────────────────────────────────────────────────
LIMB_COLOR   = (200, 220,  60)
JOINT_COLOR  = (0,   210, 255)
THREAT_COLOR = (0,    50, 230)
AMBER        = (0,   180, 255)
TEAL         = (180, 200,  80)
GREY         = (150, 150, 150)
FONT         = cv2.FONT_HERSHEY_SIMPLEX


# ── EMA Smoother ──────────────────────────────────────────────────────────────
class EMASmoother:
    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
        self._s: dict[int, list[float]] = {}

    def smooth(self, tid: int, vals: list[float]) -> list[float]:
        if tid not in self._s:
            self._s[tid] = vals[:]
        else:
            p = self._s[tid]
            if len(p) != len(vals):
                self._s[tid] = vals[:]
            else:
                self._s[tid] = [self.alpha*v+(1-self.alpha)*q for v,q in zip(vals,p)]
        return self._s[tid]

    def evict(self, active: set[int]) -> None:
        for k in [k for k in self._s if k not in active]:
            del self._s[k]


# ── Shared frame buffer (from niga2.py) ───────────────────────────────────────
class LatestFrame:
    def __init__(self):
        self._lock  = threading.Lock()
        self._frame = None
        self._stop  = False

    def update(self, frame):
        with self._lock: self._frame = frame

    def read(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self):
        with self._lock: self._stop = True

    def stopped(self):
        with self._lock: return self._stop


def _put_latest(q: queue.Queue, item) -> None:
    try:
        if q.full(): q.get_nowait()
    except queue.Empty:
        pass
    try:
        q.put_nowait(item)
    except queue.Full:
        pass


# ── Result types ──────────────────────────────────────────────────────────────
@dataclass
class WeaponResult:
    dets: list = field(default_factory=list)  # [(smooth_box, tid, label, conf)]
    held: bool = False
    miss: int  = 0

@dataclass
class PoseResult:
    keypoints: list = field(default_factory=list)  # [np.ndarray (17,2)]


# ── Threads ───────────────────────────────────────────────────────────────────
def capture_loop(cap: cv2.VideoCapture, buf: LatestFrame) -> None:
    while not buf.stopped():
        ok, frame = cap.read()
        if not ok:
            buf.stop(); break
        buf.update(frame)
        time.sleep(0.001)


def weapon_worker(buf: LatestFrame, out_q: queue.Queue,
                  model: YOLO, ema: float, device: str) -> None:
    smoother  = EMASmoother(alpha=ema)
    last_good = None
    missed    = 0

    while not buf.stopped():
        frame = buf.read()
        if frame is None:
            time.sleep(0.005); continue

        sq = cv2.resize(frame, (DISPLAY_SIZE, DISPLAY_SIZE))
        results = model.track(
            source=sq, conf=EXIT_CONF, iou=TRACK_IOU,
            persist=True, tracker="bytetrack.yaml", verbose=False,
            device=device
        )
        r0 = results[0]
        dets = []
        active_ids = set()
        accepted = False

        if r0.boxes is not None and len(r0.boxes) > 0:
            if float(r0.boxes.conf.max()) >= ENTER_CONF:
                accepted = True; missed = 0
                for i, box in enumerate(r0.boxes.xyxy):
                    c   = float(r0.boxes.conf[i])
                    lbl = model.names[int(r0.boxes.cls[i])]
                    tid = int(r0.boxes.id[i]) if r0.boxes.id is not None else -1
                    raw = box.tolist()
                    sb  = smoother.smooth(tid, raw) if tid>=0 else raw
                    if tid>=0: active_ids.add(tid)
                    dets.append((sb, tid, lbl, c))
            else:
                missed += 1
        else:
            missed += 1

        smoother.evict(active_ids)

        if accepted:
            res = WeaponResult(dets=dets); last_good = res
        elif last_good and missed <= HOLD_FRAMES:
            res = WeaponResult(dets=last_good.dets, held=True, miss=missed)
        else:
            if missed > HOLD_FRAMES: last_good = None
            res = WeaponResult()

        _put_latest(out_q, res)


def pose_worker(buf: LatestFrame, out_q: queue.Queue,
                model: YOLO, ema: float, device: str) -> None:
    smoother = EMASmoother(alpha=ema)
    tid_counter = 0   # simple per-person index (pose model may not have persistent track IDs)

    while not buf.stopped():
        frame = buf.read()
        if frame is None:
            time.sleep(0.005); continue

        sq = cv2.resize(frame, (DISPLAY_SIZE, DISPLAY_SIZE))
        results = model.predict(
            source=sq, conf=0.30, iou=TRACK_IOU, verbose=False,
            device=device
        )
        r0 = results[0]
        kps_out = []

        if r0.keypoints is not None:
            kp_data  = r0.keypoints
            boxes_d  = r0.boxes
            kp_all   = kp_data.xy.cpu().numpy()   # (N, 17, 2)

            for i, kp_xy in enumerate(kp_all):
                tid  = int(boxes_d.id[i]) if (boxes_d is not None
                                               and boxes_d.id is not None) else i
                flat = kp_xy.flatten().tolist()
                sk   = np.array(smoother.smooth(tid, flat),
                                dtype=np.float32).reshape(17, 2)
                kps_out.append(sk)

        _put_latest(out_q, PoseResult(keypoints=kps_out))


# ── Cinematic ─────────────────────────────────────────────────────────────────
def _make_vignette(s):
    cx=cy=s//2; Y,X=np.ogrid[:s,:s]
    d=np.sqrt((X-cx)**2+(Y-cy)**2)/(s*0.7)
    return (1.0-np.clip(d**1.8,0,1)).astype(np.float32)

def _make_scanlines(s):
    m=np.ones((s,s),dtype=np.float32); m[::2,:]=0.83; return m

def _build_luts():
    x=np.arange(256,dtype=np.float32)
    return (np.clip(x*0.80+22,0,255).astype(np.uint8),
            np.clip(x*0.88+8, 0,255).astype(np.uint8),
            np.clip((x/255)**0.85*245,0,255).astype(np.uint8))

_VIG=_make_vignette(DISPLAY_SIZE); _SL=_make_scanlines(DISPLAY_SIZE)
_BLUT,_GLUT,_RLUT=_build_luts()

def apply_cinematic(frame):
    b,g,r=cv2.split(frame)
    frame=cv2.merge([cv2.LUT(b,_BLUT),cv2.LUT(g,_GLUT),cv2.LUT(r,_RLUT)])
    f32=frame.astype(np.float32)*_VIG[:,:,np.newaxis]*_SL[:,:,np.newaxis]
    return np.clip(f32+np.random.normal(0,5,f32.shape),0,255).astype(np.uint8)


# ── Drawing ───────────────────────────────────────────────────────────────────
def _glow_line(img, p1, p2, color, thickness=2):
    ov=np.zeros_like(img)
    cv2.line(ov,p1,p2,color,thickness+6,cv2.LINE_AA)
    cv2.addWeighted(img,1.0,cv2.GaussianBlur(ov,(11,11),0),0.7,0,img)
    cv2.line(img,p1,p2,color,thickness,cv2.LINE_AA)

def _glow_circle(img,center,radius,color):
    ov=np.zeros_like(img)
    cv2.circle(ov,center,radius+4,color,-1,cv2.LINE_AA)
    cv2.addWeighted(img,1.0,cv2.GaussianBlur(ov,(9,9),0),0.6,0,img)
    cv2.circle(img,center,radius,color,-1,cv2.LINE_AA)

def draw_skeleton(frame, kp_xy):
    for a,b in SKELETON:
        xa,ya=kp_xy[a]; xb,yb=kp_xy[b]
        if xa>1 and ya>1 and xb>1 and yb>1:
            _glow_line(frame,(int(xa),int(ya)),(int(xb),int(yb)),LIMB_COLOR)
    for x,y in kp_xy:
        if x>1 and y>1:
            _glow_circle(frame,(int(x),int(y)),4,JOINT_COLOR)

def draw_bracket_box(frame, box, tid, label, conf, t, held=False):
    x1,y1,x2,y2=(int(v) for v in box)
    w,h=x2-x1,y2-y1; arm=max(14,min(w,h)//4)
    pulse=0.55+0.45*math.sin(t*(4 if held else 7))
    col=tuple(int(c*pulse) for c in ((0,180,180) if held else THREAT_COLOR))
    amb=tuple(int(c*pulse) for c in AMBER)
    for (cx,cy),(dx,dy) in [((x1,y1),(1,1)),((x2,y1),(-1,1)),
                              ((x1,y2),(1,-1)),((x2,y2),(-1,-1))]:
        _glow_line(frame,(cx,cy),(cx+dx*arm,cy),col)
        _glow_line(frame,(cx,cy),(cx,cy+dy*arm),col)
    mx,my=(x1+x2)//2,(y1+y2)//2
    for off in range(-18,19,12):
        cv2.line(frame,(mx+off,my),(mx+off+8,my),amb,1,cv2.LINE_AA)
        cv2.line(frame,(mx,my+off),(mx,my+off+8),amb,1,cv2.LINE_AA)
    tag=(f"[HOLD] #{tid:02d} {label.upper()} {conf:.0%}"
         if held else f"#{tid:02d} {label.upper()} {conf:.0%}")
    (tw,th),bl=cv2.getTextSize(tag,FONT,0.52,1)
    cv2.rectangle(frame,(x1,y1-th-bl-6),(x1+tw+4,y1),(0,0,0),-1)
    cv2.putText(frame,tag,(x1+2,y1-bl-2),FONT,0.52,AMBER,1,cv2.LINE_AA)

def draw_hud(frame, fps, body_count, weapon_count, held, t):
    h,w=frame.shape[:2]; arm=42
    for (cx,cy),(dx,dy) in [((8,8),(1,1)),((w-8,8),(-1,1)),
                              ((8,h-8),(1,-1)),((w-8,h-8),(-1,-1))]:
        cv2.line(frame,(cx,cy),(cx+dx*arm,cy),TEAL,2,cv2.LINE_AA)
        cv2.line(frame,(cx,cy),(cx,cy+dy*arm),TEAL,2,cv2.LINE_AA)
    cv2.putText(frame,time.strftime("  %Y-%m-%d  %H:%M:%S"),(54,28),FONT,0.55,TEAL,1,cv2.LINE_AA)
    if int(t*2)%2==0:
        cv2.circle(frame,(w-60,20),6,(0,0,200),-1,cv2.LINE_AA)
        cv2.putText(frame,"REC",(w-48,25),FONT,0.5,(0,0,200),1,cv2.LINE_AA)
    cv2.putText(frame,f"FPS {fps:5.1f}",(54,h-38),FONT,0.52,GREY,1,cv2.LINE_AA)
    cv2.putText(frame,f"BODIES: {body_count:02d}",(54,h-18),FONT,0.52,LIMB_COLOR,1,cv2.LINE_AA)
    tc=AMBER if weapon_count==0 else THREAT_COLOR
    tlbl=f"THREATS: {weapon_count:02d}"+("  [HOLD]" if held else "")
    (tw,_),_=cv2.getTextSize(tlbl,FONT,0.58,1)
    cv2.putText(frame,tlbl,(w-tw-54,h-18),FONT,0.58,tc,1,cv2.LINE_AA)
    sl="[ SENTRY AI  //  POSE + WEAPON  //  v1.0.0 ]"
    (tw,_),_=cv2.getTextSize(sl,FONT,0.45,1)
    cv2.putText(frame,sl,((w-tw)//2,22),FONT,0.45,GREY,1,cv2.LINE_AA)


# ── Robust model loader (from niga2.py) ───────────────────────────────────────
def load_pose_model(primary: str) -> YOLO:
    candidates = [primary, "yolo11n-pose.pt", "yolov8n-pose.pt"]
    for w in candidates:
        try:
            print(f"[INFO] Loading pose model: {w}")
            m = YOLO(w)
            print(f"[INFO] Pose model loaded: {w}")
            return m
        except Exception as e:
            print(f"[WARN] Failed '{w}': {e}")
            if "PytorchStreamReader" in str(e) or "central directory" in str(e):
                print(f"      → '{w}' looks corrupted. Delete it and rerun.")
    raise RuntimeError("Could not load any pose model.")


# ── Main ──────────────────────────────────────────────────────────────────────
def run(weapon_weights: str, pose_weights: str,
        camera: int, ema: float, device: str) -> None:

    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    wp = Path(weapon_weights)
    if not wp.exists():
        print(f"[ERROR] Weapon weights not found: {wp}"); sys.exit(1)

    print(f"[INFO] Loading weapon model: {wp}")
    weapon_model = YOLO(str(wp))
    pose_model   = load_pose_model(pose_weights)

    print(f"[INFO] Opening camera {camera}")
    cap = cv2.VideoCapture(camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {camera}"); sys.exit(1)

    buf      = LatestFrame()
    weapon_q = queue.Queue(maxsize=1)
    pose_q   = queue.Queue(maxsize=1)

    threading.Thread(target=capture_loop,  args=(cap, buf),                    daemon=True).start()
    threading.Thread(target=weapon_worker, args=(buf, weapon_q, weapon_model, ema, device), daemon=True).start()
    threading.Thread(target=pose_worker,   args=(buf, pose_q,   pose_model,   ema, device), daemon=True).start()

    latest_w   = WeaponResult()
    latest_p   = PoseResult()
    prev_time  = time.time()
    fps        = 0.0
    t0         = time.time()

    print("[INFO] Running — press Q to quit")
    while True:
        try: latest_w = weapon_q.get_nowait()
        except queue.Empty: pass
        try: latest_p = pose_q.get_nowait()
        except queue.Empty: pass

        frame = buf.read()
        if frame is None:
            time.sleep(0.005); continue

        display = cv2.resize(frame, (DISPLAY_SIZE, DISPLAY_SIZE))
        t = time.time() - t0

        # Draw skeletons first (underneath weapons)
        for kp in latest_p.keypoints:
            draw_skeleton(display, kp)

        # Draw weapon markers on top
        for (box, tid, lbl, conf) in latest_w.dets:
            draw_bracket_box(display, box, tid, lbl, conf, t, latest_w.held)

        display = apply_cinematic(display)
        draw_hud(display, fps, len(latest_p.keypoints),
                 len(latest_w.dets), latest_w.held, t)

        now = time.time()
        fps = 0.9*fps + 0.1/max(now-prev_time, 1e-6)
        prev_time = now

        cv2.imshow("SENTRY // POSE + WEAPON DETECTION", display)
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
    p.add_argument("--weapon-weights", default="releases/v1.0.0/best.pt")
    p.add_argument("--pose-weights",   default="releases/v1.0.0/yolo26n-pose.pt")
    p.add_argument("--camera",         type=int,   default=0)
    p.add_argument("--ema",            type=float, default=0.4)
    p.add_argument("--device",         type=str,   default="", help="Device to run on, e.g. 'cuda:0' or 'cpu'")
    return p.parse_args()

if __name__ == "__main__":
    a = parse_args()
    run(a.weapon_weights, a.pose_weights, a.camera, a.ema, a.device)
