import cv2 as cv
from net.base.DetectionNet import DetectionNet


class LandmarkDetectionNet(DetectionNet):
    def detect(self, frame, shift: tuple = (0, 0)):
        dots = []
        out = self.process(frame)

        for dot in out.reshape(int(out.shape[1]/2), 2):
            x = int(dot[0] * frame.shape[1] + shift[0])
            y = int(dot[1] * frame.shape[0] + shift[1])
            dots.append((x, y))

        self.last_output = dots
        return self.last_output

    def draw(self, frame):
        for x, y in self.last_output:
            cv.circle(frame, (x, y), 3, (255, 0, 255), -1)
