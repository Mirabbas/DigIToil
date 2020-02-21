import cv2
import numpy as np
from time import time
from math import sqrt
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default=0,
                help="path to input image")
ap.add_argument("-net", default='body')
args = vars(ap.parse_args())

# Select net type
nettype = args['net']
dist_trigger = 0.3

if nettype == 'face':
    net = cv2.dnn.readNet('face-detection-retail-0005/FP32/face-detection-retail-0005.xml',
                          'face-detection-retail-0005/FP32/face-detection-retail-0005.bin')
    lm_net = cv2.dnn.readNet('landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml',
                             'landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.bin')
    renet = cv2.dnn.readNet('face-reidentification-retail-0095/FP32/face-reidentification-retail-0095.xml',
                            'face-reidentification-retail-0095/FP32/face-reidentification-retail-0095.bin')
    netsize = (300, 300)
    renetsize = (128, 128)
    lm_netsize = (48, 48)
    trigger = 0.5
elif nettype == 'body':
    net = cv2.dnn.readNet('person-detection-retail-0013/FP32/person-detection-retail-0013.xml',
                          'person-detection-retail-0013/FP32/person-detection-retail-0013.bin')
    renet = cv2.dnn.readNet('person-reidentification-retail-0079/FP32/person-reidentification-retail-0079.xml',
                            'person-reidentification-retail-0079/FP32/person-reidentification-retail-0079.bin')
    netsize = (544, 320)
    renetsize = (64, 160)
    trigger = 0.8

    f_net = cv2.dnn.readNet('face-detection-retail-0005/FP32/face-detection-retail-0005.xml',
                          'face-detection-retail-0005/FP32/face-detection-retail-0005.bin')
    f_lm_net = cv2.dnn.readNet('landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml',
                             'landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.bin')
    f_renet = cv2.dnn.readNet('face-reidentification-retail-0095/FP32/face-reidentification-retail-0095.xml',
                            'face-reidentification-retail-0095/FP32/face-reidentification-retail-0095.bin')
    f_netsize = (300, 300)
    f_renetsize = (128, 128)
    f_lm_netsize = (48, 48)
    f_trigger = 0.5
# Specify target device.
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

renet.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
renet.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def landmark(chip, xmin, ymin):
    lm_coord.clear()
    lm_blob = cv2.dnn.blobFromImage(chip, size=lm_netsize, ddepth=cv2.CV_8U)
    lm_net.setInput(lm_blob)
    lm_out = lm_net.forward()
    lm_out = lm_out.reshape(10)
    for coords in lm_out.reshape(5,2):
        x = int(coords[0] * chip.shape[1] + xmin)
        y = int(coords[1] * chip.shape[0] + ymin)
        lm_coord.append((x, y))
       
    return lm_coord

def compare(data, chip, save=0):
    reblob = cv2.dnn.blobFromImage(chip, size=renetsize, ddepth=cv2.CV_8U)
    renet.setInput(reblob)
    reout = renet.forward()
    reout = reout.reshape(256)
    reout /= sqrt(np.dot(reout, reout))
    ide = 1
    distance = -1
    if len(data) != 0:
        for x in data:
            distance = np.dot(reout, data[x])
            ide += 1
            if distance > dist_trigger:
                ide = x
                break

    if distance < dist_trigger:
        data['id{}'.format(ide)] = reout
        if save:
            cv2.imwrite('photos/id{}.jpg'.format(ide), chip)

    return distance, ide


# Read an image.
stream = cv2.VideoCapture(args["image"])
fps = 1
chips = []
data = {}
lm_coord = []
distance = 0
times = {}
ID = 0

while True:
    start = time()
    grab, frame = stream.read()
    if not grab:
        raise Exception('Image not found!')
    # Prepare input blob and perform an inference.
    blob = cv2.dnn.blobFromImage(frame, size=netsize, ddepth=cv2.CV_8U)
    net.setInput(blob)
    out = net.forward()
    # Draw detected faces on the frame.
    objects = 0
    for detection in out.reshape(-1, 7):
        confidence = float(detection[2])
        xmin = int(detection[3] * frame.shape[1])
        ymin = int(detection[4] * frame.shape[0])
        xmax = int(detection[5] * frame.shape[1])
        ymax = int(detection[6] * frame.shape[0])

        if confidence > trigger:
            objects += 1
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            chip = frame[ymin:ymax, xmin:xmax]                
            try:
                if nettype == 'face':
                    lm_coord = landmark(chip, xmin, ymin)
                
                distance, ID = compare(data, chip, 1)
            except:
                continue
            if ID in times:
                times[ID] += 1 / fps
            else:
                times[ID] = 1 / fps
            
            for dot in lm_coord:
                cv2.circle(frame, dot, 5, (0, 0, 255), -1)

                
            cv2.putText(frame, '{}'.format(ID), (xmin, ymax - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            cv2.putText(frame, '{:.1f}s'.format(times[ID]), (xmin, ymax + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    print("distance = ", distance)

    end = time()
    fps = 1 / (end - start)
    cv2.putText(frame, 'fps:{:.2f}'.format(fps), (5, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, 'objects:{}'.format(objects), (5, frame.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, 'IDs:{}'.format(len(data)), (5, frame.shape[0]-45),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Output', cv2.resize(frame, (round(frame.shape[1]/1.3), round(frame.shape[0]/1.3))))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
