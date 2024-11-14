import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hand_detector = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

IMAGE_DIR = './dataset'

features = []
class_labels = []
for class_folder in os.listdir(IMAGE_DIR):
    for image_file in os.listdir(os.path.join(IMAGE_DIR, class_folder)):
        feature_data = []
        x_coordinates = []
        y_coordinates = []

        image = cv2.imread(os.path.join(IMAGE_DIR, class_folder, image_file))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = hand_detector.process(image_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_coordinates.append(x)
                    y_coordinates.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    feature_data.append(x - min(x_coordinates))
                    feature_data.append(y - min(y_coordinates))

            features.append(feature_data)
            class_labels.append(class_folder)

# Save features and labels into a pickle file
try:
    with open('processed_data.pickle', 'wb') as file:
        pickle.dump({'features': features, 'labels': class_labels}, file)
    print("Feature extraction complete. Data saved to 'processed_data.pickle'.")
except Exception as e:
    print(f"An error occurred while saving the pickle file: {e}")

