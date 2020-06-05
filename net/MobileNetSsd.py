import cv2 as cv
from net.base.DetectionNet import DetectionNet


class MobileNetSsd(DetectionNet):
    def detect(self, frame, confidence: float, class_id=15):
        objects = {}
        out = self.process(frame)

        for detection in out.reshape(-1, 7):
            idx = int(detection[1])
            if (idx != class_id) & (idx > 0):
                continue

            if (detection[2] > confidence):
                xmin = int(detection[3] * frame.shape[1])
                ymin = int(detection[4] * frame.shape[0])
                xmax = int(detection[5] * frame.shape[1])
                ymax = int(detection[6] * frame.shape[0])

                if objects.get(idx):
                    objects[idx].append((xmin, ymin, xmax, ymax))
                else:
                    objects[idx] = [(xmin, ymin, xmax, ymax)]

        if len(objects):
            if (class_id == -1):
                self.last_output = objects
            else:
                self.last_output = objects[class_id]
        else:
            self.last_output = []

        return self.last_output

    def draw(self, frame):
        for xmin, ymin, xmax, ymax in self.last_output:
            cv.rectangle(frame, (xmin, ymin),
                         (xmax, ymax), (255, 255, 0), 2)
