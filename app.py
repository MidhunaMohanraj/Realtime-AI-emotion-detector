import cv2
from deepface import DeepFace
import numpy as np
import time
import sys
import os

# ── Colours for each emotion label ──────────────────────────────────────────
EMOTION_COLORS = {
    "happy":     (0,   220,  80),
    "sad":       (200,  80,  30),
    "angry":     (0,    0,  230),
    "surprise":  (0,   200, 230),
    "fear":      (140,   0, 180),
    "disgust":   (0,   160,  60),
    "neutral":   (160, 160, 160),
}

EMOJI_MAP = {
    "happy":    "😄",
    "sad":      "😢",
    "angry":    "😡",
    "surprise": "😮",
    "fear":     "😨",
    "disgust":  "🤢",
    "neutral":  "😐",
}

def draw_rounded_rect(img, pt1, pt2, color, radius=12, thickness=2):
    """Draw a rounded rectangle on the frame."""
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.line(img,  (x1+radius, y1), (x2-radius, y1), color, thickness)
    cv2.line(img,  (x1+radius, y2), (x2-radius, y2), color, thickness)
    cv2.line(img,  (x1, y1+radius), (x1, y2-radius), color, thickness)
    cv2.line(img,  (x2, y1+radius), (x2, y2-radius), color, thickness)
    cv2.ellipse(img, (x1+radius, y1+radius), (radius,radius), 180,  0,  90, color, thickness)
    cv2.ellipse(img, (x2-radius, y1+radius), (radius,radius), 270,  0,  90, color, thickness)
    cv2.ellipse(img, (x1+radius, y2-radius), (radius,radius),  90,  0,  90, color, thickness)
    cv2.ellipse(img, (x2-radius, y2-radius), (radius,radius),   0,  0,  90, color, thickness)

def draw_confidence_bars(frame, emotions: dict, x: int, y: int):
    """Draw a mini bar chart of all emotion confidences."""
    bar_w_max = 120
    bar_h     = 10
    gap       = 14
    for i, (emo, score) in enumerate(sorted(emotions.items(), key=lambda e: -e[1])):
        color   = EMOTION_COLORS.get(emo, (200, 200, 200))
        bar_len = int(score / 100 * bar_w_max)
        by      = y + i * gap
        # background track
        cv2.rectangle(frame, (x, by), (x + bar_w_max, by + bar_h), (40, 40, 40), -1)
        # filled bar
        if bar_len > 0:
            cv2.rectangle(frame, (x, by), (x + bar_len, by + bar_h), color, -1)
        # label
        cv2.putText(frame, f"{emo[:7]:<7} {score:5.1f}%",
                    (x + bar_w_max + 6, by + bar_h - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1)

def draw_hud(frame, fps: float, frame_count: int, saved_count: int):
    """Overlay top-left HUD: FPS, frame count, saved screenshots."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (230, 70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    cv2.putText(frame, f"FPS: {fps:.1f}",
                (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 1)
    cv2.putText(frame, f"Frames: {frame_count}",
                (14, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
    cv2.putText(frame, f"Saved: {saved_count}",
                (14, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    # Controls legend — bottom-left
    legend = ["[S] Screenshot", "[R] Reset stats", "[Q/ESC] Quit"]
    for i, txt in enumerate(legend):
        cv2.putText(frame, txt,
                    (10, h - 10 - (len(legend) - 1 - i) * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1)

def main():
    os.makedirs("screenshots", exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check your camera is connected.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    print("=" * 55)
    print("  🎭  Realtime Emotion Detector  |  DeepFace + OpenCV")
    print("=" * 55)
    print("  Controls:")
    print("    S       → save screenshot")
    print("    R       → reset session stats")
    print("    Q / ESC → quit")
    print("=" * 55)

    # State
    result_cache   = None          # last successful DeepFace result
    frame_count    = 0
    saved_count    = 0
    analyze_every  = 3             # run DeepFace every N frames (perf balance)
    fps            = 0.0
    t_prev         = time.time()

    # Rolling emotion history for smoothing (last 5 detections)
    history        = []
    HISTORY_LEN    = 5

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        frame_count += 1

        # ── FPS calc ──────────────────────────────────────────────────────
        t_now = time.time()
        fps   = 1.0 / max(t_now - t_prev, 1e-6)
        t_prev = t_now

        # ── Run DeepFace every N frames ───────────────────────────────────
        if frame_count % analyze_every == 0:
            try:
                results = DeepFace.analyze(
                    frame,
                    actions=["emotion"],
                    enforce_detection=False,
                    silent=True,
                )
                result_cache = results if isinstance(results, list) else [results]
                # Keep rolling history
                if result_cache:
                    top_emo = result_cache[0]["dominant_emotion"]
                    history.append(top_emo)
                    if len(history) > HISTORY_LEN:
                        history.pop(0)
            except Exception:
                pass  # just keep last result

        # ── Draw detections ───────────────────────────────────────────────
        display = frame.copy()

        if result_cache:
            for face in result_cache:
                region   = face.get("region", {})
                emotions = face.get("emotion", {})
                dominant = face.get("dominant_emotion", "neutral")
                color    = EMOTION_COLORS.get(dominant, (160, 160, 160))

                # Face bounding box
                fx = region.get("x", 0)
                fy = region.get("y", 0)
                fw = region.get("w", 0)
                fh = region.get("h", 0)

                if fw > 0 and fh > 0:
                    draw_rounded_rect(display, (fx, fy), (fx+fw, fy+fh), color, radius=10, thickness=2)

                    # Semi-transparent label background
                    label_h = 36
                    overlay2 = display.copy()
                    cv2.rectangle(overlay2,
                                  (fx, fy - label_h - 4),
                                  (fx + 220, fy),
                                  (0, 0, 0), -1)
                    cv2.addWeighted(overlay2, 0.55, display, 0.45, 0, display)

                    # Dominant emotion text
                    conf = emotions.get(dominant, 0)
                    label = f"{dominant.upper()}  {conf:.0f}%"
                    cv2.putText(display, label,
                                (fx + 6, fy - 12),
                                cv2.FONT_HERSHEY_DUPLEX, 0.65, color, 1)

                    # Confidence bars (right side of face box)
                    bar_x = fx + fw + 12
                    bar_y = fy
                    if bar_x + 250 < display.shape[1]:
                        draw_confidence_bars(display, emotions, bar_x, bar_y)

        # ── Smoothed emotion banner (top-centre) ─────────────────────────
        if history:
            # Most common in recent history
            smooth_emo = max(set(history), key=history.count)
            emoji      = EMOJI_MAP.get(smooth_emo, "")
            banner     = f"{emoji}  {smooth_emo.upper()}"
            bw, bh     = 280, 40
            bx         = (display.shape[1] - bw) // 2
            ov3 = display.copy()
            cv2.rectangle(ov3, (bx, 6), (bx + bw, 6 + bh), (0, 0, 0), -1)
            cv2.addWeighted(ov3, 0.5, display, 0.5, 0, display)
            b_color = EMOTION_COLORS.get(smooth_emo, (200, 200, 200))
            cv2.putText(display, banner,
                        (bx + 10, 6 + bh - 10),
                        cv2.FONT_HERSHEY_DUPLEX, 0.75, b_color, 2)

        # ── HUD ───────────────────────────────────────────────────────────
        draw_hud(display, fps, frame_count, saved_count)

        cv2.imshow("🎭 Emotion Detector  |  Press Q to quit", display)

        # ── Key handling ──────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):          # Q or ESC
            break
        elif key == ord("s"):              # Screenshot
            ts   = time.strftime("%Y%m%d_%H%M%S")
            path = f"screenshots/emotion_{ts}.jpg"
            cv2.imwrite(path, display)
            saved_count += 1
            print(f"[INFO] Screenshot saved → {path}")
        elif key == ord("r"):              # Reset
            history.clear()
            frame_count = 0
            saved_count = 0
            print("[INFO] Stats reset.")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Session ended.")

if __name__ == "__main__":
    main()
