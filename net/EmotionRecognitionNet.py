import cv2 as cv
from net.base.DetectionNet import DetectionNet


class EmotionRecognitionNet(DetectionNet):
    EMOTIONS = ('neutral', 'happy', 'sad', 'surprise', 'anger')

    def detect(self, frame):
        out = self.process(frame)

        temp = out[0].reshape(-1, 5)[0]

        self.last_output = {
            EmotionRecognitionNet.EMOTIONS[i]: temp[i] for i in range(len(temp))}
        return self.last_output

    def draw(self, frame, org: tuple):
        emotion = self.get_dominated_emotion()
        xmin, ymin, xmax, ymax = org
        scale = ((xmax - xmin) / 150)

        if scale < 0.5:
            scale = 0.7

        elif scale > 1:
            scale = 1

        weight = int((xmax - xmin) / 50)

        if weight > 2:
            weight = 2

        cv.putText(frame, f"{emotion}", (xmax+5, ymin + 30),
                   cv.FONT_HERSHEY_COMPLEX, scale, (255, 0, 255), weight)

    def get_dominated_emotion(self):
        return max(self.last_output, key=lambda k: self.last_output[k])
