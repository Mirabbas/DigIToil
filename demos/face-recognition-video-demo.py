import argparse

import cv2
import myCV
import net

parser = argparse.ArgumentParser(description='Person recognition from video')
parser.add_argument('--video', type=str, help='video path', default=0)
args = parser.parse_args()

# Define networks
FACE_NET = net.face_detection()
FACE_REID_NET = net.face_reid()
LANDMARK_NET = net.face_lanmark()

# Define input video stream
STREAM = cv2.VideoCapture(args.video)
counter = cv2.TickMeter()
frames = 0

# Dictionary for face descriptors
face_database = {}

while True:
    counter.stop()
    counter.start()

    grab, frame = STREAM.read()
    if not grab:
        raise Exception('Image not found')

    # Copy the input image for continuing processing
    img = frame.copy()

    # Detecting faces from the input image
    faces = myCV.detect(FACE_NET, frame, 0.7)
    for fxmin, fymin, fxmax, fymax in faces:
        cv2.rectangle(img, (fxmin, fymin), (fxmax, fymax), (255, 255, 0), 2)

        # Crop face area from frame
        fcrop = frame[fymin:fymax, fxmin:fxmax]

        # Search and draw facial landmarks
        dots = myCV.get_landmarks(LANDMARK_NET, fcrop, (fxmin, fymin))
        for x, y in dots:
            cv2.circle(img, (x, y), 3, (255, 0, 255), -1)

        # Transfrom the face to improve recognition
        fcrop = myCV.face_transform(fcrop)

        # Getting facial descriptor
        face_features = myCV.get_descriptor(FACE_REID_NET, fcrop)

        # Searching the face in face_database
        ID, distance = myCV.object_in_database(
            face_features, face_database, 0.5)

        # If our database dont contain the face add it
        if not ID:
            ID = len(face_database) + 1
            face_database[ID] = face_features

        cv2.putText(img, f"[id{ID}][{(1 - distance):.2f}]", (fxmin, fymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, ((fxmax - fxmin)/150), (255, 0, 255), 2)

    # Runtime and Afps specs
    runtime = counter.getTimeSec()
    if runtime > 0:
        frames += 1
        fps = frames / runtime
        cv2.putText(img, f"[AFps:{fps:.1f}][Runtime:{runtime:.0f}s]", (5, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    cv2.imshow('Output', img)
    if cv2.waitKey(1) == ord('q'):
        break

STREAM.release()
cv2.destroyAllWindows()
