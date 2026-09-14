import cv2
import mediapipe as mp
import numpy as np
import math
import time
import os
import sys
import subprocess

TARGET_REPS        = 20
DOWN_ANGLE_THRESH  = 110
UP_ANGLE_THRESH    = 145
IDLE_TIMEOUT       = 5.0
STARTUP_GRACE      = 10.0
REP_COOLDOWN       = 0.5
MOTIVATION_VIDEO   = "motivation.mp4"

C_BG_CARD  = (20,  15,  40)
C_ACCENT   = (0,  220, 130)
C_WARN     = (0,  100, 255)
C_WHITE    = (255, 255, 255)
C_YELLOW   = (0,  220, 255)
C_PURPLE   = (200,  80, 255)
C_SKELETON = (0,  255, 180)

BODY_CONNECTIONS = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]


def calc_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return math.degrees(math.acos(np.clip(cos, -1.0, 1.0)))


def lm_px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)


def rounded_rect(img, x, y, w, h, r, color, alpha=0.82):
    ov = img.copy()
    cv2.rectangle(ov, (x+r, y), (x+w-r, y+h), color, -1)
    cv2.rectangle(ov, (x, y+r), (x+w, y+h-r), color, -1)
    for cx, cy in [(x+r,y+r),(x+w-r,y+r),(x+r,y+h-r),(x+w-r,y+h-r)]:
        cv2.circle(ov, (cx, cy), r, color, -1)
    cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)


def draw_skeleton(img, landmarks, w, h):
    for s, e in BODY_CONNECTIONS:
        ls, le = landmarks[s], landmarks[e]
        if ls.visibility > 0.3 and le.visibility > 0.3:
            p1 = lm_px(ls, w, h)
            p2 = lm_px(le, w, h)
            cv2.line(img, p1, p2, (0, 70, 50), 10)
            cv2.line(img, p1, p2, C_SKELETON, 3)


def draw_counter_card(img, count, stage, idle_frac, grace_left, angle_val, target=TARGET_REPS):
    cx, cy, cw, ch = 15, 15, 210, 180
    rounded_rect(img, cx, cy, cw, ch, 18, C_BG_CARD)

    cv2.putText(img, "PUSH-UPS", (cx+18, cy+30),
                cv2.FONT_HERSHEY_DUPLEX, 0.62, C_ACCENT, 1, cv2.LINE_AA)

    s  = str(count)
    sc = 2.8 if count < 10 else 2.2
    ts = cv2.getTextSize(s, cv2.FONT_HERSHEY_DUPLEX, sc, 3)[0]
    tx = cx + (cw - ts[0]) // 2
    cv2.putText(img, s, (tx, cy+97),
                cv2.FONT_HERSHEY_DUPLEX, sc, C_WHITE, 3, cv2.LINE_AA)
    cv2.putText(img, s, (tx, cy+97),
                cv2.FONT_HERSHEY_DUPLEX, sc, C_ACCENT, 1, cv2.LINE_AA)
    cv2.putText(img, f"/ {target}", (cx+cw-55, cy+97),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_PURPLE, 1, cv2.LINE_AA)

    bx, by_, bw, bh = cx+18, cy+108, cw-36, 6
    cv2.rectangle(img, (bx, by_), (bx+bw, by_+bh), (60,60,90), -1)
    fill = int(bw * min(count / target, 1.0))
    if fill > 0:
        cv2.rectangle(img, (bx, by_), (bx+fill, by_+bh), C_ACCENT, -1)
    cv2.rectangle(img, (bx, by_), (bx+bw, by_+bh), (100,100,140), 1)

    if grace_left > 0:
        iy = by_ + 11
        cv2.rectangle(img, (bx, iy), (bx+bw, iy+bh), (30,50,30), -1)
        gfill = int(bw * (1.0 - grace_left / STARTUP_GRACE))
        if gfill > 0:
            cv2.rectangle(img, (bx, iy), (bx+gfill, iy+bh), C_YELLOW, -1)
        cv2.putText(img, f"Get ready: {int(grace_left)+1}s", (cx+18, cy+143),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_YELLOW, 1, cv2.LINE_AA)
    else:
        iy = by_ + 11
        cv2.rectangle(img, (bx, iy), (bx+bw, iy+bh), (60,30,30), -1)
        idle_fill = int(bw * min(idle_frac, 1.0))
        if idle_fill > 0:
            cv2.rectangle(img, (bx, iy), (bx+idle_fill, iy+bh), C_WARN, -1)

    sc_col = C_ACCENT if stage == "UP" else C_WARN
    cv2.putText(img, stage if stage else "---", (cx+18, cy+162),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, sc_col, 2, cv2.LINE_AA)

    a_col = C_WARN if angle_val < DOWN_ANGLE_THRESH else (C_ACCENT if angle_val > UP_ANGLE_THRESH else C_WHITE)
    cv2.putText(img, f"Angle: {int(angle_val)}", (cx+100, cy+162),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, a_col, 1, cv2.LINE_AA)


def overlay_video_on_frame(frame, mot_frame, W, H):
    ov_w = int(W * 0.90)
    ov_h = int(ov_w * mot_frame.shape[0] / mot_frame.shape[1])
    ov_h = min(ov_h, int(H * 0.90))
    ov_w = int(ov_h * mot_frame.shape[1] / mot_frame.shape[0])
    resized = cv2.resize(mot_frame, (ov_w, ov_h))
    x_off = (W - ov_w) // 2
    y_off = (H - ov_h) // 2
    roi = frame[y_off:y_off+ov_h, x_off:x_off+ov_w]
    cv2.addWeighted(resized, 0.92, roi, 0.08, 0, roi)
    frame[y_off:y_off+ov_h, x_off:x_off+ov_w] = roi
    cv2.rectangle(frame, (x_off-2, y_off-2), (x_off+ov_w+2, y_off+ov_h+2), (0,100,255), 3)


def draw_done(img, w, h):
    ov = img.copy()
    cv2.rectangle(ov, (0,0),(w,h),(0,30,10),-1)
    cv2.addWeighted(ov, 0.55, img, 0.45, 0, img)
    items = [
        ("GOAL ACHIEVED!", C_ACCENT, 1.5),
        (f"{TARGET_REPS} PUSH-UPS DONE!", C_YELLOW, 0.9),
        ("Press  Q  to quit", C_WHITE, 0.55),
    ]
    y = h // 2 - 70
    for text, col, scale in items:
        ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 2)[0]
        cv2.putText(img, text, ((w-ts[0])//2, y),
                    cv2.FONT_HERSHEY_DUPLEX, scale, col, 2, cv2.LINE_AA)
        y += int(ts[1] * 2.6)


def find_video(default):
    if os.path.isfile(default):
        return default
    sd = os.path.dirname(os.path.abspath(__file__))
    t = os.path.join(sd, default)
    if os.path.isfile(t):
        return t
    for f in sorted(os.listdir(sd)):
        if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            return os.path.join(sd, f)
    return None


def main():
    mot_path = MOTIVATION_VIDEO
    for i, a in enumerate(sys.argv[1:]):
        if a in ("--video", "-v") and i+1 < len(sys.argv[1:]):
            mot_path = sys.argv[i+2]
            break
    mot_path = find_video(mot_path)

    pose = mp.solutions.pose.Pose(
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mot_cap    = cv2.VideoCapture(mot_path) if mot_path else None
    mot_fps    = (mot_cap.get(cv2.CAP_PROP_FPS) or 30) if mot_cap else 30
    cached_mot = None

    count            = 0
    stage            = None
    goal_done        = False
    start_time       = time.time()
    last_rep_time    = time.time()
    last_count_time  = 0.0
    mot_showing      = False
    last_mot_frame_t = time.time()
    audio_proc       = None
    stop_grace_start = None
    live_angle       = 160.0

    win_name = "Push-Up Counter"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, W, H)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        now        = time.time()
        elapsed    = now - start_time
        grace_left = max(0.0, STARTUP_GRACE - elapsed)
        idle       = now - last_rep_time

        lms = results.pose_landmarks

        if lms:
            sl = lms.landmark
            draw_skeleton(frame, sl, W, H)

            if not goal_done:
                def pt(i):
                    return [sl[i].x, sl[i].y]

                la_vis = min(sl[11].visibility, sl[13].visibility, sl[15].visibility)
                ra_vis = min(sl[12].visibility, sl[14].visibility, sl[16].visibility)

                use_angle = None

                if la_vis > 0.3:
                    la = calc_angle(pt(11), pt(13), pt(15))
                    use_angle = la
                if ra_vis > 0.3:
                    ra = calc_angle(pt(12), pt(14), pt(16))
                    if use_angle is not None:
                        use_angle = (use_angle + ra) / 2
                    else:
                        use_angle = ra

                if use_angle is not None:
                    live_angle = use_angle
                    print(f"\rAngle: {use_angle:6.1f}  Stage: {stage or '---':4s}  Count: {count}", end="", flush=True)

                    if use_angle <= DOWN_ANGLE_THRESH:
                        if stage != "DOWN":
                            stage = "DOWN"

                    if use_angle >= UP_ANGLE_THRESH and stage == "DOWN":
                        if (now - last_count_time) >= REP_COOLDOWN:
                            stage = "UP"
                            count += 1
                            last_rep_time   = now
                            last_count_time = now

        should_play = (grace_left <= 0) and (idle >= IDLE_TIMEOUT) and not goal_done and mot_cap is not None

        if should_play:
            stop_grace_start = None
            if not mot_showing:
                mot_showing      = True
                last_mot_frame_t = now
                if mot_path:
                    audio_proc = subprocess.Popen(
                        ["afplay", mot_path],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
            if (now - last_mot_frame_t) >= (1.0 / mot_fps):
                ret_m, mf = mot_cap.read()
                if not ret_m:
                    mot_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret_m, mf = mot_cap.read()
                    if audio_proc:
                        audio_proc.terminate()
                    if mot_path:
                        audio_proc = subprocess.Popen(
                            ["afplay", mot_path],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                        )
                if ret_m:
                    cached_mot       = mf
                    last_mot_frame_t = now
            if cached_mot is not None:
                overlay_video_on_frame(frame, cached_mot, W, H)
        else:
            if mot_showing:
                if stop_grace_start is None:
                    stop_grace_start = now
                elif (now - stop_grace_start) >= 0.4:
                    mot_showing      = False
                    stop_grace_start = None
                    cached_mot       = None
                    if audio_proc:
                        audio_proc.terminate()
                        audio_proc = None
                    if mot_cap:
                        mot_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if count >= TARGET_REPS and not goal_done:
            goal_done  = True
            cached_mot = None
            if audio_proc:
                audio_proc.terminate()
                audio_proc = None

        draw_counter_card(frame, count, stage, idle / IDLE_TIMEOUT, grace_left, live_angle)
        if goal_done:
            draw_done(frame, W, H)

        cv2.imshow(win_name, frame)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break

    if audio_proc:
        audio_proc.terminate()
    if mot_cap:
        mot_cap.release()
    cap.release()
    cv2.destroyAllWindows()
    pose.close()


if __name__ == "__main__":
    main()
