import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN

# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# create the detector, using default weights
detector = MTCNN()


def get_faces_with_eye(faces, eyes, margin=0.8):
    """
    Filter faces by removing faces with no detected eye
    Faces and eyes is described a box. This function removes faces where there is no eye box inside

    Arguments:
    faces -- numpy array containing boxes for detected faces (nb faces, 4)
        [X0, Y0, weight, length]
    eyes -- numpy array containing boxes for detected eyes (nb eyes, 4)
        [X0, Y0, weight, length]
    Return:
    faces_with_eye -- numpy array containing boxes for detected faces containing at least one eye (nb faces, 4)
    """

    faces_with_eye = np.array([]).reshape((0, 4))

    # Loop over faces
    for i_face in faces:

        # Extract the coordinates
        (x_f, y_f, w_f, h_f) = i_face

        # Init detect eye in face
        is_eye_in_face = False

        # Loop over eyes
        for (x_e, y_e, w_e, h_e) in eyes:

            # Accept X% of eye in face
            # So add padding to the face box
            x_f_margin = x_f - (1 - margin)*w_e
            y_f_margin = y_f - (1 - margin)*h_e
            w_f_margin = w_f + 2*(1 - margin)*w_e
            h_f_margin = h_f + 2*(1 - margin)*h_e

            # Detect eye in face
            if x_e > x_f_margin and\
                y_e > y_f_margin and\
                (x_e + w_e) < (x_f_margin + w_f_margin) and\
                (y_e + h_e) < (y_f_margin + h_f_margin):
                is_eye_in_face = True

        # Keep only faces with eye
        if is_eye_in_face:
            faces_with_eye = np.append(faces_with_eye, i_face.reshape(1, 4), axis=0)

    return faces_with_eye


def detect_face_opencv(frame):
    """
    Detect faces in a frame

    Arguments:
    frame -- RGB image (X, Y , 3)
    Return:
    faces_with_eye -- numpy array containing boxes for detected faces containing at least one eye (nb faces, 4)
    """

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flipped = cv2.flip(frame, cv2.COLOR_BGR2GRAY)

    # Detect the faces (and profiles)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    profiles_left = profile_cascade.detectMultiScale(gray, 1.1, 4)
    profiles_right = profile_cascade.detectMultiScale(flipped, 1.1, 4)
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)

    # Concatenate faces
    faces_list = np.array([]).reshape((0, 4))
    if type(faces) is not tuple:
        faces_list = np.append(faces_list, faces, axis=0)
    if type(profiles_left) is not tuple:
        faces_list = np.append(faces_list, profiles_left, axis=0)
    if type(profiles_right) is not tuple:
        profiles_right[:, 0] = gray.shape[1] - (profiles_right[:, 0] + profiles_right[:, 2])  # Flip the X coordinates
        faces_list = np.append(faces_list, profiles_right, axis=0)

    # Filter faces by eyes
    faces_with_eye = get_faces_with_eye(faces_list, eyes)

    # Convert float into integer
    faces_with_eye = np.array(faces_with_eye, dtype=int)

    return faces_with_eye


def detect_face_mtcnn(frame):
    """
    Detect faces in a frame

    Arguments:
    frame -- RGB image (X, Y , 3)
    Return:
    faces -- numpy array containing boxes for detected faces (nb faces, 4)
    """

    # Detect the faces
    boxes = detector.detect_faces(frame)

    # Convert the faces into a matrix
    faces = np.array([]).reshape((0, 4))

    # Extract the face only if there is one
    if boxes:
        for box in boxes:
            faces = np.append(faces, np.array(box['box']).reshape((1,4)), axis=0)

    # Convert float into integer
    faces = np.array(faces, dtype=int)

    return faces


if __name__ == '__main__':
    # define a video capture object
    vid = cv2.VideoCapture(0)

    while True:

        # Capture the video frame
        # by frame
        _, frame = vid.read()

        faces = detect_face_mtcnn(frame)

        i = 0
        # Draw the rectangle around each face
        # if len(faces) > 0:
        for (x, y, w, h) in faces:
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
