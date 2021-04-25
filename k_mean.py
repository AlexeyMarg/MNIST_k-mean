class k_mean:
    def __init__(self, k, dimension, max_deviation):
        self.k = k
        self.dimension = dimension
        self.max_deviation = max_deviation
        self.stored_data = []
        self.nearest_vectors = []
        for i in range(k):
            self.nearest_vectors.append([self.max_deviation, -1])

    def load_data(self, path):
        pass

    def rmse(self, loaded_vector, predict_vector):
        pass

    def run(self, predict_vector):
        pass
