import numpy as np
from math import sqrt, pow

class k_mean:
    def __init__(self, k, dimension, max_deviation):
        self.k = k
        self.dimension = dimension
        self.max_deviation = max_deviation
        self.stored_data = np.empty((0, 0))
        self.nearest_vectors = np.zeros((k, 2), float)
        for i in range(k):
            self.nearest_vectors[i][0] = 255
            self.nearest_vectors[i][1] = -1

    def load_data(self, path):
        data = []
        train_set_file = open(path, 'r')
        while True:
            s = train_set_file.readline()
            if len(s) == 0:
                break
            data.append(s.split(','))
        train_set_file.close()
        self.stored_data = np.asfarray(data)

    def rmse(self, loaded_vector, predict_vector):
        summ = 0
        for i in (loaded_vector[1:] - predict_vector):
            summ += pow(i, 2)
        return sqrt(summ / self.dimension)

    def run_single(self, predict_vector):
        for i in self.stored_data:
            deviation = self.rmse(i.transpose(), predict_vector)
            self.nearest_vectors = np.append(self.nearest_vectors, [[deviation, i[0]]], axis=0)
            self.nearest_vectors = self.nearest_vectors[np.argsort(self.nearest_vectors[:,0])]
            self.nearest_vectors  = self.nearest_vectors[:-1]

        possible_classes = []
        for i in self.nearest_vectors:
            if i[1] not in possible_classes and i[1] != -1:
                possible_classes.append(i[1])
        for i in range(len(possible_classes)):
            possible_classes[i] = [possible_classes[i], 0]
        for i in self.nearest_vectors:
            for j in possible_classes:
                if i[1] == j[0]:
                    j[1] += 1

        for i in possible_classes:
            i[0], i[1] = i[1], i[0]

        return int(sorted(possible_classes)[0][1])


