import os
import cv2

SAVE_DIR = './dataset'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

class_count = 36
images_per_class = 100

camera = cv2.VideoCapture(0)

for class_index in range(class_count):
    class_path = os.path.join(SAVE_DIR, str(class_index))
    if not os.path.exists(class_path):
        os.makedirs(class_path)

    print(f'Collecting data for class {class_index}')

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Failed to capture image. Check camera connection.")
            break

        frame = cv2.flip(frame, 1)

        cv2.putText(frame, 'Ready? Press "S" to start or "Q" to quit.', (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        key = cv2.waitKey(25)
        if key == ord('s'):
            break
        elif key == ord('q'):
            camera.release()
            cv2.destroyAllWindows()
            exit()

    # Start collecting images for the current class
    image_index = 0
    while image_index < images_per_class:
        ret, frame = camera.read()
        if not ret:
            print("Failed to capture image. Check camera connection.")
            break

        # Flip the frame horizontally for consistency
        frame = cv2.flip(frame, 1)

        # Show the frame and save the image
        cv2.imshow('frame', frame)
        image_path = os.path.join(class_path, f'{image_index}.jpg')
        cv2.imwrite(image_path, frame)
        image_index += 1

        # Press 'Q' to quit the program
        if cv2.waitKey(25) == ord('q'):
            camera.release()
            cv2.destroyAllWindows()
            exit()

camera.release()
cv2.destroyAllWindows()
