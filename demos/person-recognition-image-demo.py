import argparse

import cv2
import myCV
import net

parser = argparse.ArgumentParser(description='Person recognition from image')
parser.add_argument('img', type=str, help='Image path')
args = parser.parse_args()

# Define networks
FACE_NET = net.face_detection()
LANDMARK_NET = net.face_lanmark()
FACE_REID_NET = net.face_reid()
BODY_NET = net.mobilenet_ssd()

# Dictionary for face descriptors
face_database = {}

frame = cv2.imread(args.img)
cv2.imshow('Input', frame)
cv2.waitKey(0)

# Copy the input image for continuing processing
img = frame.copy()

n = 0
# Detect bodies from the input image
# 15 = person id in mobilessd list
bodies = myCV.detect(BODY_NET, frame, 0.7, 15)
for bxmin, bymin, bxmax, bymax in bodies:
    cv2.rectangle(img, (bxmin, bymin), (bxmax, bymax), (255, 255, 0), 2)

    # Crop body area from frame
    bcrop = frame[bymin:bymax, bxmin:bxmax]

    n += 1
    cv2.imshow('Person{}'.format(n), bcrop)
    cv2.waitKey(0)

    # Detect faces from the the body crop image
    face = myCV.detect(FACE_NET, bcrop, 0.7)
    for fxmin, fymin, fxmax, fymax in face:
        fxmin += bxmin
        fymin += bymin
        fxmax += bxmin
        fymax += bymin

        cv2.rectangle(img, (fxmin, fymin),
                      (fxmax, fymax), (0, 255, 0), 2)

        fcrop = frame[fymin:fymax, fxmin:fxmax]

        cv2.imshow('Face{}'.format(n), fcrop)
        cv2.waitKey(0)

        # Search and draw facial landmarks
        dots = myCV.get_landmarks(LANDMARK_NET, fcrop, (fxmin, fymin))
        for x, y in dots:
            cv2.circle(img, (x, y), 4, (255, 0, 255), -1)

        face_features = myCV.get_descriptor(FACE_REID_NET, fcrop)

        # Search face in face data dict
        ID, distance = myCV.object_in_database(
            face_features, face_database, 0.5)

        # If our database dont contain the face add it
        if not ID:
            ID = len(face_database) + 1
            face_database[ID] = face_features

        cv2.putText(img, f'id{ID}', (bxmin, bymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

cv2.imshow('Output', img)
cv2.waitKey(0)
