from myCV import Net

INTEL_MODELS_DIR = 'D:/YandexDisk/Projects/OpenVINO/models/intel/'


# Face detection net
def face_detection():
    name = 'face-detection-retail-0005'
    path = INTEL_MODELS_DIR + name + '/FP16/'
    prototxt = path + name + '.xml'
    weight = path + name + '.bin'
    size = (300, 300)

    net = Net(prototxt, weight, size)
    return net


# Landmarks detection net
def face_lanmark():
    path = 'D:/YandexDisk/Projects/OpenVINO/models/intel/landmarks-regression-retail-0009/FP16/'
    prototxt = path + 'landmarks-regression-retail-0009.xml'
    weight = path + 'landmarks-regression-retail-0009.bin'
    size = (48, 48)

    net = Net(prototxt, weight, size)
    return net


# Face reidification net
def face_reid():
    path = 'D:/YandexDisk/Projects/OpenVINO/models/intel/face-reidentification-retail-0095/FP16/'
    prototxt = path + 'face-reidentification-retail-0095.xml'
    weight = path + 'face-reidentification-retail-0095.bin'
    size = (128, 128)

    net = Net(prototxt, weight, size)
    return net


# Person detection net
def person_detection():
    path = 'D:/YandexDisk/Projects/OpenVINO/models/intel/person-detection-retail-0013/FP16/'
    prototxt = path + 'person-detection-retail-0013.xml'
    weight = path + 'person-detection-retail-0013.bin'
    size = (544, 320)

    net = Net(prototxt, weight, size)
    return net


# Person reidification net
def person_reid():
    path = 'D:/YandexDisk/Projects/OpenVINO/models/intel/person-reidentification-retail-0079/FP16/'
    prototxt = path + 'person-reidentification-retail-0079.xml'
    weight = path + 'person-reidentification-retail-0079.bin'
    size = (64, 160)

    net = Net(prototxt, weight, size)
    return net
