import os
import cv2
import mediapipe as mp
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

dataset_path = r"D:\IEEE Sign language Recognition\Gesture Image Data\Gesture Image Data"
model_path = "sign_model.pkl"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)

def extract_landmarks(image):
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        landmarks = []
        for lm in results.multi_hand_landmarks[0].landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        return landmarks
    return None

def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape(-1, 3)
    base_x, base_y, base_z = landmarks[0]
    landmarks -= [base_x, base_y, base_z]
    max_val = np.max(np.abs(landmarks))
    if max_val > 0:
        landmarks /= max_val
    return landmarks.flatten()

X, y = [], []
for label in os.listdir(dataset_path):
    folder = os.path.join(dataset_path, label)
    if not os.path.isdir(folder):
        continue
    for file in os.listdir(folder):
        img_path = os.path.join(folder, file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        landmarks = extract_landmarks(img)
        if landmarks:
            landmarks = normalize_landmarks(landmarks)
            X.append(landmarks)
            y.append(label)

X = np.array(X)
y = np.array(y)

print(f" Loaded dataset: {len(X)} samples, {len(set(y))} classes")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

save = SVC(kernel='linear', probability=True, class_weight='balanced')
save.fit(X_train, y_train)

print(" Training accuracy:", save.score(X_train, y_train))
print(" Test accuracy:", save.score(X_test, y_test))

with open(model_path, "wb") as f:
    pickle.dump(save, f)

print(" Model trained and saved as:", model_path)
