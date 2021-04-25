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
        for i in (loaded_vector - predict_vector):
            summ += pow(i, 2)
        return sqrt(summ / self.dimension)

    def run(self, predict_vector):
        pass
