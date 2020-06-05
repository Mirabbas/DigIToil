from net.base.NeuralNet import NeuralNet


class RecognitionNet(NeuralNet):
    def get_descriptor(self, frame):
        self.last_output = self.process(frame)
        return self.last_output
