import cv2
import mediapipe as mp
import csv

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

label = input("Enter gesture label: ")

with open("dataset/gestures.csv", "a", newline="") as f:
    writer = csv.writer(f)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        if results.multi_hand_landmarks:

            for hand in results.multi_hand_landmarks:

                landmarks = []

                for lm in hand.landmark:
                    landmarks.append(lm.x)
                    landmarks.append(lm.y)

                landmarks.append(label)

                writer.writerow(landmarks)

                mp_draw.draw_landmarks(
                    frame,
                    hand,
                    mp_hands.HAND_CONNECTIONS
                )

        cv2.imshow("Collecting Data", frame)

        if cv2.waitKey(1) == 27:
            break

cap.release()
cv2.destroyAllWindows()
