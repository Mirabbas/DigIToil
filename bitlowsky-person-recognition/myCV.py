import numpy as np
import cv2
from scipy.spatial.distance import cosine


class Net():
    def __init__(self, model_prototxt, model_weight, size: tuple, scale=1):
        self.net = cv2.dnn.readNet(model_prototxt, model_weight)
        self.scale = scale
        self.size = size

        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)

    def process(self, img):
        try:
            blob = cv2.dnn.blobFromImage(
                img, self.scale, self.size, ddepth=cv2.CV_8U)
        except:
            return [[0 for x in range(7)]]
        else:
            self.net.setInput(blob)
            return self.net.forward()


def detect(net: Net, img, conf: float, ID: int = 1):
    objects = {}
    out = net.process(img)

    for detection in out.reshape(-1, 7):
        idx = int(detection[1])
        if (idx != ID) & (idx > 0):
            continue

        confidence = detection[2]
        if (confidence > conf):
            xmin = int(detection[3] * img.shape[1])
            ymin = int(detection[4] * img.shape[0])
            xmax = int(detection[5] * img.shape[1])
            ymax = int(detection[6] * img.shape[0])

            if objects.get(idx):
                objects[idx].append((xmin, ymin, xmax, ymax))
            else:
                objects[idx] = [(xmin, ymin, xmax, ymax)]

    if len(objects):
        if (ID == -1):
            return objects
        else:
            return objects[ID]
    else:
        return []


def get_landmarks(net: Net, img, shift: tuple = (0, 0)):
    dots = []
    out = net.process(img)

    for dot in out.reshape(int(out.shape[1]/2), 2):
        x = int(dot[0] * img.shape[1] + shift[0])
        y = int(dot[1] * img.shape[0] + shift[1])
        dots.append((x, y))

    return dots


def get_descriptor(net: Net, img):
    return net.process(img)


def get_distance(descriptor_1, descriptor_2):
    return cosine(descriptor_1, descriptor_2)


def object_in_database(descriptor, data: dict, distance_trigger: float):
    min_distance = 1
    ID = 0
    if len(data) != 0:
        for x in data:
            distance = get_distance(descriptor, data[x])
            if distance >= distance_trigger:
                continue
            elif distance < min_distance:
                min_distance = distance
                ID = x
    return ID, min_distance


def face_transform(img):
    return img
