import cv2
import numpy as np

# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')


def detect_face_opencv(frame):

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flipped = cv2.flip(frame, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    profiles_left = profile_cascade.detectMultiScale(gray, 1.1, 4)
    profiles_right = profile_cascade.detectMultiScale(flipped, 1.1, 4)

    faces_list = np.array([]).reshape((0, 4))
    if type(faces) is not tuple:
        faces_list = np.append(faces_list, faces, axis=0)
    if type(profiles_left) is not tuple:
        faces_list = np.append(faces_list, profiles_left, axis=0)
    if type(profiles_right) is not tuple:
        profiles_right[:, 1] = gray.shape[1] - profiles_right[:, 1]  # Flip the X coordinates
        faces_list = np.append(faces_list, profiles_right, axis=0)

    faces_list = np.array(faces_list, dtype=int)
    return faces_list


if __name__ == '__main__':
    # define a video capture object
    vid = cv2.VideoCapture(0)

    while True:

        # Capture the video frame
        # by frame
        _, frame = vid.read()

        faces = detect_face_opencv(frame)

        i = 0
        # Draw the rectangle around each face
        # if len(faces) > 0:
        for (x, y, w, h) in faces:
            # cv2.rectangle(frame, (0, 0), (100, 100), (255, 0, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
