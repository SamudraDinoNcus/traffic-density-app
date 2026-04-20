import numpy as np

class TrafficPredictor:
    def __init__(self, model):
        self.model = model
        self.history = []

    def moving_average(self, data, window=10):
        if len(data) < window:
            return np.mean(data)
        return np.mean(data[-window:])

    def extract_features(self, count):
        self.history.append(count)

        count_smooth = self.moving_average(self.history)
        density = count_smooth
        flow = count - self.history[-2] if len(self.history) > 1 else 0
        delta = flow
        var = np.var(self.history[-10:]) if len(self.history) >= 10 else 0

        return [count_smooth, density, flow, delta, var]

    def predict(self, count):
        features = self.extract_features(count)
        return self.model.predict([features])[0]