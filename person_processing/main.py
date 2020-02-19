import cv2
import myCV
import net


face_net = net.face_detection()
landmark_net = net.face_lanmark()
face_reid_net = net.face_reid()
body_net = myCV.Net("mo_mobilenet-ssd.xml", "mo_mobilenet-ssd.bin", (300, 300))


stream = cv2.VideoCapture(0)
counter = cv2.TickMeter()
faces_data = {}


while True:
    counter.stop()
    counter.start()

    grab, frame = stream.read()
    if not grab:
        raise Exception('Image not found')

    img = frame.copy()

    # 15 = person id in mobilessd list
    bodies = myCV.detect(body_net, frame, 0.7, 15)
    for bxmin, bymin, bxmax, bymax in bodies:
        cv2.rectangle(img, (bxmin, bymin), (bxmax, bymax), (255, 255, 0), 2)

        bchip = frame[bymin:bymax, bxmin:bxmax]

        face = myCV.detect(face_net, bchip, 0.7)
        for fxmin, fymin, fxmax, fymax in face:
            fxmin += bxmin
            fymin += bymin
            fxmax += bxmin
            fymax += bymin

            cv2.rectangle(img, (fxmin, fymin),
                          (fxmax, fymax), (0, 255, 0), 2)

            fchip = frame[fymin:fymax, fxmin:fxmax]

            dots = myCV.get_landmarks(landmark_net, fchip, (fxmin, fymin))
            for x, y in dots:
                cv2.circle(img, (x, y), 3, (255, 0, 255), -1)

            face_info = myCV.get_descriptor(face_reid_net, fchip)

            ID = myCV.object_in_data(face_info, faces_data, 0.5)
            if not ID:
                ID = len(faces_data) + 1
            # Rewriting data!!!
            faces_data[ID] = face_info

            cv2.putText(img, 'id{}'.format(ID), (bxmin, bymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    runtime = counter.getTimeSec()
    cv2.putText(img, 'Runtime:{:.0f}s'.format(runtime), (5, img.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    cv2.imshow('Output', img)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
        stream.release()
        cv2.destroyAllWindows()
