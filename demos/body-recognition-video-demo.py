import argparse

import cv2
import myCV
import net

parser = argparse.ArgumentParser(description='Person recognition from video')
parser.add_argument('--video', type=str, help='video path', default=0)
args = parser.parse_args()

# Define networks
BODY_NET = net.mobilenet_ssd()
BODY_REID_NET = net.person_reid()

# Define input video stream
STREAM = cv2.VideoCapture(args.video)
counter = cv2.TickMeter()
frames = 0

# Dictionary for body descriptors
bodies_database = {}

while True:
    counter.stop()
    counter.start()

    grab, frame = STREAM.read()
    if not grab:
        raise Exception('Frame not found')

    # Copy the input image for continuing processing
    img = frame.copy()

    # Detect bodies from the input image
    # 15 = person id in mobilessd list
    bodies = myCV.detect(BODY_NET, frame, 0.7, 15)
    for bxmin, bymin, bxmax, bymax in bodies:
        cv2.rectangle(img, (bxmin, bymin), (bxmax, bymax), (255, 255, 0), 2)

        # Crop body area from frame
        bcrop = frame[bymin:bymax, bxmin:bxmax]

        # Getting body descriptor
        body_features = myCV.get_descriptor(BODY_REID_NET, bcrop)

        # Searching the body in body_database
        ID, distance = myCV.object_in_database(
            body_features, bodies_database, 0.5)

        # If our database dont contain the body add it
        if not ID:
            ID = len(bodies_database) + 1
            bodies_database[ID] = body_features

        # Display ID confidence of person
        cv2.putText(img, f"[id{ID}][{(1 - distance):.2f}]", (bxmin, bymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

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
