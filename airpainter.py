import cv2
import mediapipe as mp
import numpy as np
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)


canvas = None
prev_x, prev_y = 0, 0

colors = [(255,0,0),(0,255,0),(0,0,255),(0,255,255),(255,0,255)]
color_index = 0
current_color = colors[color_index]

brush_size = 5
eraser_size = 40

last_action_time = 0
cooldown = 2

save_hold_start = None
save_delay = 2

while cap.isOpened():

    success, frame = cap.read()
    if not success:
        break

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    h,w,_ = frame.shape
    action_text = "Idle"

    for i,color in enumerate(colors):
        cv2.rectangle(frame,(i*60,0),(i*60+60,60),color,-1)

    if results.multi_hand_landmarks:

        hands_data = []

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)

            index_up = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
            middle_up = hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y
            ring_up = hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y
            pinky_up = hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y

            thumb_up = hand_landmarks.landmark[4].y < hand_landmarks.landmark[3].y
            thumb_down = hand_landmarks.landmark[4].y > hand_landmarks.landmark[3].y

            palm_up = index_up and middle_up and ring_up and pinky_up
            fist = not (index_up or middle_up or ring_up or pinky_up)

            ix = int(hand_landmarks.landmark[8].x*w)
            iy = int(hand_landmarks.landmark[8].y*h)

            hands_data.append({
                "index": index_up,
                "middle": middle_up,
                "pinky": pinky_up,
                "thumb_up": thumb_up,
                "thumb_down": thumb_down,
                "palm": palm_up,
                "fist": fist,
                "x": ix,
                "y": iy
            })

        current_time = time.time()

        if len(hands_data) == 2 and hands_data[0]["fist"] and hands_data[1]["fist"]:

            action_text = "Hold to Save PNG"

            if save_hold_start is None:
                save_hold_start = current_time

            elif current_time - save_hold_start > save_delay:

                filename = f"drawing_{int(time.time())}.png"
                cv2.imwrite(filename, canvas)

                action_text = f"Saved: {filename}"
                save_hold_start = None

        else:

            save_hold_start = None

            hand = hands_data[0]
            ix,iy = hand["x"],hand["y"]

            if hand["palm"]:
                cv2.circle(canvas,(ix,iy),eraser_size,(0,0,0),-1)
                action_text = "Erase"

            elif hand["pinky"] and not hand["index"]:
                action_text = "Change Color"

                if current_time - last_action_time > cooldown:
                    color_index = (color_index+1)%len(colors)
                    current_color = colors[color_index]
                    last_action_time = current_time

            elif hand["thumb_up"] and not hand["index"]:
                action_text = f"Brush + ({brush_size})"

                if current_time - last_action_time > cooldown:
                    brush_size += 2
                    brush_size = min(30,brush_size)
                    last_action_time = current_time

            elif hand["thumb_down"] and not hand["index"]:
                action_text = f"Brush - ({brush_size})"

                if current_time - last_action_time > cooldown:
                    brush_size -= 2
                    brush_size = max(2,brush_size)
                    last_action_time = current_time

            elif hand["index"] and hand["middle"]:
                prev_x,prev_y = ix,iy
                action_text = "Move"

            elif hand["index"]:

                if prev_x==0 and prev_y==0:
                    prev_x,prev_y = ix,iy

                cx = int((prev_x+ix)/2)
                cy = int((prev_y+iy)/2)

                cv2.line(canvas,(prev_x,prev_y),(cx,cy),current_color,brush_size)

                prev_x,prev_y = cx,cy
                action_text = "Drawing"

            else:
                prev_x,prev_y = 0,0

    frame = cv2.add(frame,canvas)

    cv2.putText(frame,f"Mode: {action_text}",(200,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

    cv2.circle(frame,(520,30),brush_size,current_color,-1)

    cv2.imshow("AI Gesture Drawing Board",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()