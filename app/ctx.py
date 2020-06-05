import net

face_detection_net = net.FaceDetectionNet(
    "models/face-detection-retail-0005.xml",
    "models/face-detection-retail-0005.bin",
    (300, 300)
)

landmark_detection_net = net.LandmarkDetectionNet(
    "models/landmarks-regression-retail-0009.xml",
    "models/landmarks-regression-retail-0009.bin",
    (48, 48)
)

age_gender_recognition_net = net.AgeGenderRecognitionNet(
    "models/age-gender-recognition-retail-0013.xml",
    "models/age-gender-recognition-retail-0013.bin",
    (62, 62)
)

emotion_recognition_net = net.EmotionRecognitionNet(
    "models/emotions-recognition-retail-0003.xml",
    "models/emotions-recognition-retail-0003.bin",
    (64, 64)
)

head_pose_estimation_net = net.HeadPoseEstimationNet(
    "models/head-pose-estimation-adas-0001.xml",
    "models/head-pose-estimation-adas-0001.bin",
    (60, 60)
)

mobilenet_ssd = net.MobileNetSsd(
    "models/mo_mobilenet-ssd.xml",
    "models/mo_mobilenet-ssd.bin",
    (300, 300)
)

face_reidentification_net = net.RecognitionNet(
    "models/face-reidentification-retail-0095.xml",
    "models/face-reidentification-retail-0095.bin",
    (128, 128)
)

person_reidentification_net = net.RecognitionNet(
    "models/person-reidentification-retail-0200.xml",
    "models/person-reidentification-retail-0200.bin",
    (128, 256)
)

person_detection_net = net.PersonDetectionNet(
    "models/person-detection-retail-0013.xml",
    "models/person-detection-retail-0013.bin",
    (544, 320)
)
