import cv2
import numpy as np
import mediapipe as mp
from pynput.keyboard import Controller, Key

kb = Controller()

# ---- Key mapping (edit as per your game) ----
KEY_LEFT  = 'a'      # or Key.left
KEY_RIGHT = 'd'      # or Key.right
KEY_ACCEL = 'w'      # or Key.up
KEY_BRAKE = 's'      # or Key.down
KEY_CLUTCH = Key.shift

# ---- thresholds (tune) ----
TOUCH_ON  = 0.35   # smaller = stricter
TOUCH_OFF = 0.45   # hysteresis release
STEER_DEADZONE_DEG = 8

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

def dist(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))

def lm_xy(lm, w, h):
    return (int(lm.x * w), int(lm.y * h))

class KeyState:
    def __init__(self):
        self.down = set()

    def press(self, k):
        if k not in self.down:
            kb.press(k); self.down.add(k)

    def release(self, k):
        if k in self.down:
            kb.release(k); self.down.remove(k)

    def release_all(self):
        for k in list(self.down):
            kb.release(k)
        self.down.clear()

ks = KeyState()

# Touch state (for hysteresis)
acc_touch = False
brk_touch = False
clu_touch = False

cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    # Collect hands
    hand_data = []
    if res.multi_hand_landmarks:
        for hand_lms in res.multi_hand_landmarks:
            lms = hand_lms.landmark
            wrist = lm_xy(lms[0], w, h)
            hand_data.append((wrist[0], lms))

    # If less than 2 hands, safe release
    if len(hand_data) < 2:
        ks.release_all()
        cv2.putText(frame, "Need 2 hands for wheel", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow("Gesture Racing", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    # Sort hands left->right by wrist x
    hand_data.sort(key=lambda x: x[0])
    left_lms = hand_data[0][1]
    right_lms = hand_data[1][1]

    # ---- Steering (KEEP YOUR LOGIC) ----
    # Using wrist points as example:
    L = lm_xy(left_lms[0], w, h)   # left wrist
    R = lm_xy(right_lms[0], w, h)  # right wrist

    cv2.line(frame, L, R, (255, 180, 80), 4)
    cv2.circle(frame, L, 10, (255,180,80), -1)
    cv2.circle(frame, R, 10, (255,180,80), -1)

    angle = np.degrees(np.arctan2((R[1]-L[1]), (R[0]-L[0])))  # -180..180
    # For steering left/right we only need sign after deadzone:
    steer = 0
    if angle > STEER_DEADZONE_DEG:
        steer = 1   # right
    elif angle < -STEER_DEADZONE_DEG:
        steer = -1  # left

    # Apply steering keys
    if steer == -1:
        ks.press(KEY_LEFT); ks.release(KEY_RIGHT)
        steer_txt = "LEFT"
    elif steer == 1:
        ks.press(KEY_RIGHT); ks.release(KEY_LEFT)
        steer_txt = "RIGHT"
    else:
        ks.release(KEY_LEFT); ks.release(KEY_RIGHT)
        steer_txt = "STRAIGHT"

    cv2.putText(frame, f"Steer: {steer_txt}  angle={angle:.1f}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200,200,255), 2)

    # ---- Pedals by thumb touches (NO BOX UI) ----
    # Right hand: ACC = thumb(4) touch index(8), BRK = thumb(4) touch middle(12)
    rt = lm_xy(right_lms[4], w, h)   # right thumb tip
    ri = lm_xy(right_lms[8], w, h)   # right index tip
    rm = lm_xy(right_lms[12], w, h)  # right middle tip
    rw = lm_xy(right_lms[0], w, h)   # right wrist

    # Left hand: CLU = thumb(4) touch index(8)
    lt = lm_xy(left_lms[4], w, h)
    li = lm_xy(left_lms[8], w, h)
    lw = lm_xy(left_lms[0], w, h)

    # Normalize by hand size (wrist to index_mcp)
    r_scale = max(dist(rw, lm_xy(right_lms[5], w, h)), 1.0)
    l_scale = max(dist(lw, lm_xy(left_lms[5], w, h)), 1.0)

    acc_d = dist(rt, ri) / r_scale
    brk_d = dist(rt, rm) / r_scale
    clu_d = dist(lt, li) / l_scale

    # hysteresis update
    if not acc_touch and acc_d < TOUCH_ON: acc_touch = True
    if acc_touch and acc_d > TOUCH_OFF: acc_touch = False

    if not brk_touch and brk_d < TOUCH_ON: brk_touch = True
    if brk_touch and brk_d > TOUCH_OFF: brk_touch = False

    if not clu_touch and clu_d < TOUCH_ON: clu_touch = True
    if clu_touch and clu_d > TOUCH_OFF: clu_touch = False

    # Priority: brake > accel
    if brk_touch:
        ks.press(KEY_BRAKE); ks.release(KEY_ACCEL)
    else:
        ks.release(KEY_BRAKE)
        if acc_touch:
            ks.press(KEY_ACCEL)
        else:
            ks.release(KEY_ACCEL)

    # Clutch independent
    if clu_touch:
        ks.press(KEY_CLUTCH)
    else:
        ks.release(KEY_CLUTCH)

    # UI indicators
    def indicator(x, y, label, on, color_on):
        color = color_on if on else (70,70,70)
        cv2.rectangle(frame, (x,y), (x+120,y+40), color, -1)
        cv2.putText(frame, label, (x+15,y+28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    indicator(20, 70,  "ACC", acc_touch, (40,160,80))
    indicator(150, 70, "BRK", brk_touch, (50,50,180))
    indicator(280, 70, "CLU", clu_touch, (120,80,200))

    cv2.imshow("Gesture Racing", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

ks.release_all()
cap.release()
cv2.destroyAllWindows()