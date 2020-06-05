from net.base.NeuralNet import NeuralNet


class DetectionNet(NeuralNet):
    def detect(self, frame):
        raise NotImplementedError

    def draw(self, frame):
        raise NotImplementedError
