from scipy.spatial.distance import cosine


class FeatureDataBase:
    def __init__(self):
        self.data = {}

    def add(self, descriptor):
        self.data[len(self.data) + 1] = descriptor

    def update(self, ID, descriptor):
        self.data[ID] = descriptor

    def get_distance(self, descriptor_1, descriptor_2):
        return cosine(descriptor_1, descriptor_2)

    def get_id(self, descriptor, distance_trigger: float):
        min_distance = 1
        object_id = 0

        if len(self.data):
            for x in self.data:
                distance = self.get_distance(descriptor, self.data[x])

                if distance >= distance_trigger:
                    continue

                elif distance < min_distance:
                    min_distance = distance
                    object_id = x

        return object_id
