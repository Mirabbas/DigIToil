import cv2 as cv
from net.base.DetectionNet import DetectionNet


class FaceDetectionNet(DetectionNet):
    def detect(self, frame, confidence: float):
        objects = []
        out = self.process(frame)

        for detection in out.reshape(-1, 7):
            if (detection[2] > confidence):
                xmin = int(detection[3] * frame.shape[1])
                ymin = int(detection[4] * frame.shape[0])
                xmax = int(detection[5] * frame.shape[1])
                ymax = int(detection[6] * frame.shape[0])

                objects.append((xmin, ymin, xmax, ymax))

        self.last_output = objects
        return self.last_output

    def draw(self, frame):
        for xmin, ymin, xmax, ymax in self.last_output:
            cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)
