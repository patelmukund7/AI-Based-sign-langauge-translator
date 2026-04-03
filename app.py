from flask import Flask, Response
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pandas as pd

app = Flask(__name__)

# Load model
model = load_model("model/sign_model.h5")

# Label encoder
data = pd.read_csv("dataset/gestures.csv", header=None)
labels = data.iloc[:, -1]

encoder = LabelEncoder()
encoder.fit(labels)

# Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Camera
cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmark_list = []

                for lm in hand_landmarks.landmark:
                    landmark_list.append(lm.x)
                    landmark_list.append(lm.y)

                if len(landmark_list) == 42:
                    input_data = np.array(landmark_list).reshape(1, -1)
                    prediction = model.predict(input_data)
                    class_id = np.argmax(prediction)
                    gesture = encoder.inverse_transform([class_id])[0]

                    cv2.putText(frame, f"Gesture: {gesture}", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def home():
    return """
    <h2>Sign Language AI</h2>
    <img src="/video">
    """


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
