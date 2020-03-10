from myCV import Net


# Face detection net
def face_detection():
    path = 'models/'
    prototxt = path + 'face-detection-retail-0005.xml'
    weight = path + 'face-detection-retail-0005.bin'
    size = (300, 300)

    return Net(prototxt, weight, size)


# Landmarks detection net
def face_lanmark():
    path = 'models/'
    prototxt = path + 'landmarks-regression-retail-0009.xml'
    weight = path + 'landmarks-regression-retail-0009.bin'
    size = (48, 48)

    return Net(prototxt, weight, size)


# Face reidification net
def face_reid():
    path = 'models/'
    prototxt = path + 'face-reidentification-retail-0095.xml'
    weight = path + 'face-reidentification-retail-0095.bin'
    size = (128, 128)

    return Net(prototxt, weight, size)


# Person detection net
def person_detection():
    path = 'models/'
    prototxt = path + 'person-detection-retail-0013.xml'
    weight = path + 'person-detection-retail-0013.bin'
    size = (544, 320)

    return Net(prototxt, weight, size)


# Person reidification net
def person_reid():
    path = 'models/'
    prototxt = path + 'person-reidentification-retail-0079.xml'
    weight = path + 'person-reidentification-retail-0079.bin'
    size = (64, 160)

    return Net(prototxt, weight, size)


# Mobilenet-ssd converted to IE format
def mobilenet_ssd():
    path = 'models/'
    prototxt = path + 'mo_mobilenet-ssd.xml'
    weight = path + 'mo_mobilenet-ssd.bin'
    size = (300, 300)

    return Net(prototxt, weight, size)
