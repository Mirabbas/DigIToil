import cv2 as cv


class Stream:
    def __init__(self, source=0):
        self._stream = cv.VideoCapture(source)
        self._counter = cv.TickMeter()
        self._frames = 0
        self._runtime = 0
        self._fps = 0
        self._frame_time = 0

    def get_frame(self):
        self._start_time = self._counter.getTimeSec()
        return next(self._frame_generator())

    @property
    def frames(self):
        return self._frames

    @property
    def runtime(self):
        return self._runtime

    @property
    def frame_time(self):
        self._frame_time = self._counter.getTimeSec() - self._start_time
        return self._frame_time

    @property
    def fps(self):
        if not self._fps:
            self._temp_frames = 1
            self._temp_time = self.frame_time
        else:
            self._temp_frames += 1
            self._temp_time += self.frame_time

        if self._temp_time > 0:
            self._fps = self._temp_frames / self._temp_time

        if self._temp_time >= 3:
            self._temp_time = 0
            self._temp_frames = 0

        return self._fps

    def _frame_generator(self):
        while True:
            self._counter.stop()
            self._counter.start()

            grab, frame = self._stream.read()

            if not grab:
                raise Exception('Image not found')

            self._frames += 1
            self._runtime = self._counter.getTimeSec()

            yield frame

    def get_average_fps(self):
        if self.runtime > 0:
            return self._frames / self._runtime
        else:
            return 0
