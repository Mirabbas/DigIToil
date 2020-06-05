import cv2 as cv
from net.base.DetectionNet import DetectionNet


class AgeGenderRecognitionNet(DetectionNet):
    def __init__(self, topology, weight, size: tuple, scale=1):
        super().__init__(topology, weight, size, scale)
        self.output_layers = ['prob', 'age_conv3']

    def detect(self, frame):
        out = self.process(frame)

        temp = out[0].reshape(-1, 2)[0]

        if temp[0] < temp[1]:
            gender = "male"
        else:
            gender = "female"

        age = int(out[1].reshape(-1, 1)[0][0] * 100)

        self.last_output = (age, gender)
        return self.last_output

    def draw(self, frame, org: tuple):
        age, gender = self.last_output
        xmin, ymin, xmax, ymax = org
        scale = ((xmax - xmin) / 150)

        if scale < 0.5:
            scale = 0.7

        elif scale > 1:
            scale = 1

        weight = int((xmax - xmin) / 50)

        if weight > 2:
            weight = 2

        cv.putText(frame, f"{gender}, {age}", (xmax+5, ymin + 10),
                   cv.FONT_HERSHEY_COMPLEX, scale, (255, 0, 255), weight)
