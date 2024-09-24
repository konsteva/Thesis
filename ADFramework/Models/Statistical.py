""" Statistical "models" such as simple mean and rolling mean are placed here """
import numpy as np
from tqdm import tqdm

from ADFramework.Models.Model import Model
from ADFramework.Utilities import Utils


class MeanLineModel(Model):
    """ Static mean line """
    def __init__(self):
        super().__init__("Mean Line Model")
        self.mean = None

    def fit(self, x_train, x_val=None, training_config=None, early_stopping=None):
        self.mean = np.mean(x_train)

        return self.mean

    def predict(self, x, reconstruct=False):
        pred = np.full(len(x), self.mean)
        res = np.abs(x - pred)

        return pred, res

    def save(self, save_dir, file_name):
        pass

    def load(cls, model_dir, file_name):
        pass


class RollingMeanModel(Model):
    """ Rolling mean using sliding window """
    def __init__(self, forecast_window):
        super().__init__("Rolling Mean Model")
        self.forecast_window = forecast_window

    def fit(self, x_train, x_val=None, training_config=None, early_stopping=None):
        pass

    def predict(self, x, reconstruct=False):
        x_segments = Utils.segment_timeseries(x, window=self.forecast_window)
        pred = np.zeros(shape=(len(x_segments), self.forecast_window))
        for i in tqdm(range(1, len(x_segments))):
            mean = np.mean(x_segments[i-1])
            pred[i-1] = np.full(len(x_segments[i]), mean)

        pred, confints = Utils.average_reconstruct_timeseries_confint(pred)
        res = np.abs(x[:len(pred)] - pred)

        print(self.forecast_window)
        print(x_segments.shape)
        print(x.shape)
        print(res.shape)

        return pred, confints, res

    def save(self, save_dir, file_name):
        pass

    @classmethod
    def load(cls, model_dir, file_name):
        pass
