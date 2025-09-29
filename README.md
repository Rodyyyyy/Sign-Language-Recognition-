# Sign Language Recognition System

The project is a Machine Learning and Computer Vision based system designed to recognize Sign Language letters using a webcam. It leverages MediaPipe for hand tracking and Scikit-Learn (SVM) for classification.This project is a part of the AI Committee in IEEE AIU Student Branch.

---

## Features

* Real-time hand landmark detection using [MediaPipe](https://google.github.io/mediapipe/).
* Letter prediction with probability scores.
* Converts sequential predictions into words.
* Dataset preprocessing and model training pipeline included.
* Achieves reliable performance with normalized 3D hand landmarks.

---

## Tech Stack

* Programming Language: Python
* Machine Learning: Scikit-Learn (Support Vector Machine - SVM)
* Computer Vision: OpenCV, MediaPipe 
* Model Persistence: Pickle

---

## Project Structure

```
Sign-Language-Recognition
│── Train.py          # Model training scrip
│── Sign reco.py      # Real-time recognition using webcam
│── sign_model.pkl    # Saved trained model (generated after training)
│── Gesture Image Data # Dataset (images of hand signs, structured by class folders)
│── README.md         # Project documentation
```

---

## How It Works

### 1. Training Phase (`Train.py`)

* Loads dataset images of sign language gestures.
* Extracts 21 hand landmarks (x, y, z) from each image using MediaPipe.
* Normalizes landmarks to make training invariant to position/scale.
* Trains an SVM classifier with balanced class weights.
* Saves trained model as `sign_model.pkl`.

### 2. Recognition Phase (`Sign reco.py`)

* Captures frames from your webcam.
* Detects hand landmarks in real-time.
* Normalizes the landmarks and feeds them into the trained model.
* Predicts the current letter and builds words dynamically.
* Displays predictions and confidence scores on the video feed.

---

## Example Output

* During training:

  ```
   Loaded dataset: 1500+ samples, 26 classes
   Training accuracy: 0.95
   Test accuracy: 0.91
   Model trained and saved as: sign_model.pkl
  ```

* During recognition:

  * Webcam feed with:

    * Letter predictions with confidence score.
    * Current word being formed.


* Add sentence prediction using sequence models (LSTM/Transformer).
* Deploy via web app (Flask/Streamlit) for accessibility.
