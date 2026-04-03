import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load trained model
model = load_model("model/sign_model.h5")

# Load dataset to rebuild label encoder
data = pd.read_csv("dataset/gestures.csv", header=None)
labels = data.iloc[:, -1]

encoder = LabelEncoder()
encoder.fit(labels)

# Initialize mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)


mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            landmark_list = []

            for lm in hand_landmarks.landmark:
                landmark_list.append(lm.x)
                landmark_list.append(lm.y)

            if len(landmark_list) == 42:

                input_data = np.array(landmark_list).reshape(1, -1)

                prediction = model.predict(input_data)
                class_id = np.argmax(prediction)

                gesture = encoder.inverse_transform([class_id])[0]

                cv2.putText(
                    frame,
                    f'Prediction: {gesture}',
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Sign Language Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
