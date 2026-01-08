import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

#setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cooldown = 0.7
last_action_time = 0

#volume tracking
volume_level = 0.5     #0-1
volume_smooth = 0.15

#fps smoothing
prev_time = time.time()
fps_smooth = 0
fps_alpha = 0.1

#fingers count
def count_fingers(hand):
    tips = [4, 8, 12, 16, 20]
    count = 0

    #thumb
    if hand.landmark[4].x < hand.landmark[3].x:
        count += 1

    #other fingers
    for tip in tips[1:]:
        if hand.landmark[tip].y < hand.landmark[tip - 2].y:
            count += 1

    return count

#volume meter
def draw_volume_meter(frame, level):
    h, w, _ = frame.shape
    bar_h = int(np.interp(level, [0, 1], [20, h - 80]))

    x1, y1 = w - 60, h - 40
    x2, y2 = w - 30, h - bar_h

    cv2.rectangle(frame, (x1, 40), (x2, h - 40), (60, 60, 60), 2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), -1)
    cv2.putText(frame, "VOL", (w - 75, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    
#main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    fingers = None

    if res.multi_hand_landmarks:
        hand = res.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        fingers = count_fingers(hand)

    now = time.time()

    #media actions
    if fingers is not None and (now - last_action_time) > cooldown:

        if fingers == 0:
            pyautogui.press("playpause")
            action = "Play / Pause"

        elif fingers == 1:
            pyautogui.press("volumeup")
            volume_level = min(1.0, volume_level + 0.05)
            action = "Volume Up"

        elif fingers == 2:
            pyautogui.press("volumedown")
            volume_level = max(0.0, volume_level - 0.05)
            action = "Volume Down"

        elif fingers == 3:
            pyautogui.press("nexttrack")
            action = "Next Track"

        elif fingers == 5:
            pyautogui.press("prevtrack")
            action = "Previous Track"

        else:
            action = None

        if action:
            last_action_time = now
            cv2.putText(frame, action, (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    #smooth volume visual
    volume_level += (volume_level - volume_level) * volume_smooth
    draw_volume_meter(frame, volume_level)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    fps_smooth += (fps - fps_smooth) * fps_alpha
    cv2.putText(frame, f"FPS: {int(fps_smooth)}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Gesture Media Control", frame)

    if cv2.waitKey(1) & 0xFF in (27, ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
