import cv2
import mediapipe as mp
import pickle
import numpy as np

model_path = "sign_model.pkl"
with open(model_path, "rb") as reco:
    return_model = pickle.load(reco)

mp_detect_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_detect_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape(-1, 3)
    base_x, base_y, base_z = landmarks[0]
    landmarks -= [base_x, base_y, base_z]
    max_val = np.max(np.abs(landmarks))
    if max_val > 0:
        landmarks /= max_val
    return landmarks.flatten()

cam = cv2.VideoCapture(0)

current_word = ""
words = []
hands_active = False
last_prediction = None

while cam.isOpened():
    ret, frame = cam.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        if not hands_active:
            current_word = ""
            hands_active = True

        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            landmarks = normalize_landmarks(landmarks)

            probs = return_model.predict_proba([landmarks])[0]
            confidence = max(probs)
            prediction = return_model.classes_[np.argmax(probs)]

            if confidence > 0.5:
                if prediction != last_prediction:
                    current_word += prediction
                    last_prediction = prediction

                cv2.putText(frame, f"Letter: {prediction} ({confidence:.2f})",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 0), 2, cv2.LINE_AA)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_detect_hands.HAND_CONNECTIONS)

    else:
        if hands_active:
            if current_word:
                words.append(current_word)
                print(f"Finalized Word: {current_word}")
            current_word = ""
            last_prediction = None
            hands_active = False

    h, w, _ = frame.shape
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (255, 0, 0), 5)

    cv2.putText(frame, f"Current word: {current_word}",
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("AIU IEEE Sign Language Recognition System", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()
