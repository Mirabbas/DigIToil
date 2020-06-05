import cv2 as cv


class NeuralNet:
    def __init__(self, topology, weight, size: tuple, scale=1):
        self.net = cv.dnn.readNet(topology, weight)
        self.size = size
        self.scale = scale
        self.output_layers = None
        self.last_output = None

        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_INFERENCE_ENGINE)

    def process(self, frame):
        try:
            blob = cv.dnn.blobFromImage(
                frame, self.scale, self.size, ddepth=cv.CV_8U)
            self.net.setInput(blob)

        except:
            pass

        finally:
            if self.output_layers:
                return self.net.forward(self.output_layers)
            else:
                return self.net.forward()
