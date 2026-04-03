# AI-Based Sign Language Interpreter

This project implements a real-time sign language interpreter using computer vision and deep learning. It detects hand gestures through a webcam and converts them into readable text.

---

## Project Overview

The objective of this project is to reduce the communication gap between deaf or mute individuals and others by recognizing hand gestures and displaying their meaning as text.

The system uses MediaPipe for hand landmark detection and a neural network model for gesture classification.

---

## Features

* Real-time gesture detection using a webcam
* Detection of 21 hand landmarks
* Gesture classification using a trained neural network
* Displays predicted gesture as text
* Lightweight and efficient implementation

---

## Technologies Used

* Python
* OpenCV
* MediaPipe
* TensorFlow / Keras
* NumPy
* Pandas
* Scikit-learn

---

## Project Structure

```
sign_language_ai/
│
├── collect_data.py             # Script to collect gesture data
├── train_model.py              # Script to train the model
├── predict.py                  # Real-time gesture prediction
├── gestures.csv                # Dataset of hand landmarks
├── sign_model.h5               # Trained model file
└── venv/                       # Virtual environment
```

---

## Installation and Setup

### Clone the Repository

```
git clone https://github.com/your-username/sign_language_ai.git
cd sign_language_ai
```

---

### Create Virtual Environment

```
python -m venv venv
```

Activate the environment:

Windows:

```
venv\Scripts\activate
```

Mac/Linux:

```
source venv/bin/activate
```

---

### Install Dependencies

```
pip install opencv-python mediapipe tensorflow numpy pandas scikit-learn
```

---

## Working Principle

1. The webcam captures live video input
2. MediaPipe detects hand landmarks
3. A total of 21 landmarks (x and y coordinates) are extracted
4. The feature vector is passed to a trained neural network
5. The model predicts the gesture
6. The predicted output is displayed as text on the screen

---

## Usage

### Step 1: Data Collection

```
python collect_data.py
```

Enter a gesture label (for example: Hello, Yes, No) and collect multiple samples for each gesture.

---

### Step 2: Model Training

```
python train_model.py
```

This trains the model and generates:

* model.h5

---

### Step 3: Real-Time Prediction

```
python predict.py
```

This starts the webcam and displays predicted gestures in real time.

---

## Model Details

* Input layer: 42 features (21 landmarks with x and y coordinates)
* Hidden layers: 256 and 128 neurons with ReLU activation
* Dropout layer to reduce overfitting
* Output layer with softmax activation for multi-class classification

---

## Future Improvements

* Support for dynamic gestures using sequence models
* Sentence formation from multiple gestures
* Deployment as a web or mobile application
* Support for Indian Sign Language (ISL)

---

## License

This project is available under the MIT License.

---

## Acknowledgements

This project uses MediaPipe for hand tracking, TensorFlow for model development, and OpenCV for image processing.
