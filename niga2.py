import os
import cv2
import time
import threading
import queue
from pathlib import Path
from ultralytics import YOLO


# -----------------------------
# Robust pose loader (fixes your crash)
# -----------------------------
def _file_info(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return "missing"
    size_mb = p.stat().st_size / (1024 * 1024)
    head = b""
    try:
        with p.open("rb") as f:
            head = f.read(16)
    except Exception:
        pass
    return f"exists, size={size_mb:.2f}MB, head={head!r}"


def load_pose_model():
    """
    Tries pose weights in order. If a local file is corrupted, tells you what to delete.
    """
    candidates = ["yolo11n-pose.pt", "yolov8n-pose.pt"]
    last_err = None

    for w in candidates:
        try:
            print(f"🧠 Loading pose model: {w} ...")
            m = YOLO(w)  # auto-downloads if missing (unless a broken local file blocks it)
            print(f"✅ Pose model loaded: {w}")
            return m
        except Exception as e:
            last_err = e
            info = _file_info(w)
            print(f"⚠️ Failed to load '{w}' ({info})")
            print(f"   Error: {repr(e)}")
            if "PytorchStreamReader failed reading zip archive" in str(e) or "failed finding central directory" in str(e):
                print(f"👉 '{w}' looks CORRUPTED.")
                print(f"   Fix: delete/rename '{w}' and rerun so Ultralytics can re-download it.\n")

    raise RuntimeError(f"Could not load pose model. Last error: {repr(last_err)}")


# -----------------------------
# Shared latest-frame buffer
# -----------------------------
class LatestFrame:
    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._stop = False

    def update(self, frame):
        with self._lock:
            self._frame = frame

    def read(self):
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def stop(self):
        with self._lock:
            self._stop = True

    def stopped(self):
        with self._lock:
            return self._stop


def capture_loop(cap: cv2.VideoCapture, buf: LatestFrame):
    while not buf.stopped():
        ok, frame = cap.read()
        if not ok:
            buf.stop()
            break
        buf.update(frame)
        time.sleep(0.001)


def put_latest(q: queue.Queue, item):
    """Keep only the newest item in a 1-slot queue."""
    try:
        if q.full():
            q.get_nowait()
    except queue.Empty:
        pass
    try:
        q.put_nowait(item)
    except queue.Full:
        pass


# -----------------------------
# Weapon worker: YOLO track + stabilizer
# -----------------------------
def weapon_worker(buf: LatestFrame, out_q: queue.Queue, model: YOLO, target_classes):
    ENTER_CONF = 0.35
    EXIT_CONF = 0.20
    HOLD_FRAMES = 10

    last_good = None
    missed = 0
    prev_time = 0

    while not buf.stopped():
        frame = buf.read()
        if frame is None:
            time.sleep(0.005)
            continue

        now = time.time()
#        fps = 1 / (now - prev_time) if prev_time else 0
        fps = 60  #      prev_time = now

        results = model.track(
            source=frame,
            conf=EXIT_CONF,
            iou=0.7,
            classes=target_classes,
            persist=True,
            verbose=False,
            tracker="bytetrack.yaml",
        )
        r0 = results[0]

        accepted = False
        if r0.boxes is not None and len(r0.boxes) > 0:
            confs = r0.boxes.conf
            best_idx = int(confs.argmax())
            best_conf = float(confs[best_idx])

            if best_conf >= ENTER_CONF:
                last_good = r0
                missed = 0
                accepted = True
            else:
                missed += 1
        else:
            missed += 1

        if (not accepted) and (last_good is not None) and (missed <= HOLD_FRAMES):
            annotated = last_good.plot()
            cv2.putText(
                annotated, f"WEAPON HOLD ({missed}/{HOLD_FRAMES})",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 255), 2, cv2.LINE_AA
            )
        else:
            if missed > HOLD_FRAMES:
                last_good = None
            annotated = r0.plot()

        cv2.putText(
            annotated, f"Weapon FPS: {fps:.1f}",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
            (0, 255, 0), 2, cv2.LINE_AA
        )

        put_latest(out_q, annotated)


# -----------------------------
# Pose worker: independent pose model
# -----------------------------
def pose_worker(buf: LatestFrame, out_q: queue.Queue, model: YOLO):
    prev_time = 0

    while not buf.stopped():
        frame = buf.read()
        if frame is None:
            time.sleep(0.005)
            continue

        now = time.time()
        fps = 1 / (now - prev_time) if prev_time else 0
        prev_time = now

        results = model.predict(
            source=frame,
            conf=0.25,
            iou=0.7,
            verbose=False,
        )
        r0 = results[0]
        annotated = r0.plot()

        cv2.putText(
            annotated, f"Pose FPS: {fps:.1f}",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
            (255, 255, 0), 2, cv2.LINE_AA
        )

        put_latest(out_q, annotated)


# -----------------------------
# Combine two annotated frames into one (same trail)
# -----------------------------
def blend_frames(a, b, alpha=0.55):
    """
    alpha = weight of A. B gets (1-alpha).
    """
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    # Ensure same size
    if a.shape[:2] != b.shape[:2]:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    return cv2.addWeighted(a, alpha, b, 1.0 - alpha, 0.0)


# -----------------------------
# Main combined runner
# -----------------------------
def run_combined_webcam():
    # Camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    # Preload models (fail early, not in threads)
    print("🦁 Loading weapon model: best.pt ...")
    weapon_model = YOLO("best.pt")
    print("✅ Weapon model loaded: best.pt")

    pose_model = load_pose_model()

    # Shared buffer + output queues
    buf = LatestFrame()
    weapon_q = queue.Queue(maxsize=1)
    pose_q = queue.Queue(maxsize=1)

    # Threads
    t_cap = threading.Thread(target=capture_loop, args=(cap, buf), daemon=True)
    t_weapon = threading.Thread(target=weapon_worker, args=(buf, weapon_q, weapon_model, [0]), daemon=True)
    t_pose = threading.Thread(target=pose_worker, args=(buf, pose_q, pose_model), daemon=True)

    t_cap.start()
    t_weapon.start()
    t_pose.start()

    print("✅ Combined mode: ONE window, both overlays. Press 'q' to quit. 🦁")

    weapon_frame = None
    pose_frame = None

    # Combined FPS (display loop FPS)
    prev = 0

    try:
        while True:
            # Grab latest outputs (if any)
            try:
                weapon_frame = weapon_q.get_nowait()
            except queue.Empty:
                pass
            try:
                pose_frame = pose_q.get_nowait()
            except queue.Empty:
                pass

            combined = blend_frames(weapon_frame, pose_frame, alpha=0.60)
            if combined is not None:
                now = time.time()
                fps = 1 / (now - prev) if prev else 0
                prev = now

                cv2.putText(
                    combined, f"Combined Display FPS: {fps:.1f}",
                    (20, combined.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2, cv2.LINE_AA
                )
                cv2.imshow("COMBINED: Weapon + Pose", combined)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if buf.stopped():
                break

    finally:
        buf.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("👋 Stopped cleanly.")


if __name__ == "__main__":
    run_combined_webcam()
