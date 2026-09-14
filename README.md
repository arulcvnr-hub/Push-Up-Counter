# Push-Up Counter

Real-time push-up counter using your webcam. Tracks body pose, counts reps, and plays a motivation video when you stop moving.

## What it does

- Detects your body via webcam using MediaPipe
- Draws a neon skeleton over your body
- Counts push-ups by tracking elbow angle (DOWN < 110° → UP > 145° = 1 rep)
- Shows rep count + progress bar in the top-left corner
- If you pause for **5 seconds**, a motivation video plays as an overlay
- Celebrates when you hit **20 push-ups**

## Setup

```bash
pip install -r requirements.txt
```

Place your motivation video in this folder named `motivation.mp4` (any `.mp4/.mov/.avi/.mkv` is auto-detected).

## Run

```bash
python3 pushup_counter.py
```

Press **Q** or **ESC** to quit.

## Tips

- Camera should have a clear side or front view of your arms
- Wait for the **10-second countdown** to finish before starting
- Watch the `Angle:` value in the card — it must dip below **110** on the way down
