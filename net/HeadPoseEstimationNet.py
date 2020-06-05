from math import cos, pi, sin

import cv2 as cv
from net.base.DetectionNet import DetectionNet


class HeadPoseEstimationNet(DetectionNet):
    def __init__(self, topology, weight, size: tuple, scale=1):
        super().__init__(topology, weight, size, scale)
        self.output_layers = ['angle_p_fc', 'angle_r_fc', 'angle_y_fc']

    def detect(self, frame):
        out = self.process(frame)
        p, r, y = out[0][0], out[1][0], out[2][0]

        self.last_output = (p, r, y)
        return self.last_output

    def draw(self, frame, org: tuple):
        p, r, y = self.last_output

        cos_r = cos(r * pi / 180)
        sin_r = sin(r * pi / 180)
        sin_y = sin(y * pi / 180)
        cos_y = cos(y * pi / 180)
        sin_p = sin(p * pi / 180)
        cos_p = cos(p * pi / 180)

        x = int((org[0] + org[2]) / 2)
        y = int((org[1] + org[3]) / 2)

        length = int((org[2] - org[0])/2)

        # center to right
        cv.line(frame, (x, y), (x+int(length*(cos_r*cos_y+sin_y*sin_p*sin_r)),
                                y+int(length*cos_p*sin_r)), (0, 0, 255), 2)
        # center to top
        cv.line(frame, (x, y), (x+int(length*(cos_r*sin_y*sin_p+cos_y*sin_r)),
                                y-int(length*cos_p*cos_r)), (0, 255, 0), 2)
        # center to forward
        cv.line(frame, (x, y), (x + int(length*sin_y*cos_p),
                                y + int(length*sin_p)), (255, 0, 0), 2)
