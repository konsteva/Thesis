import os
import pickle

import numpy as np

from ADFramework.Models.Model import Model
from ADFramework.Utilities import Utils
from sklearn.decomposition import PCA as _pca


class PCA(Model):
    def __init__(self, input_size=None):
        super().__init__("PCA")
        self.forecast_window = input_size
        self.num_components = None
        self.pca = None
        self.mean_ = None
        self.explained_variance = None
        self.cumulative_variance = None

    def fit(self, x_train, x_val=None, training_config=None, early_stopping=None):
        train_segments = Utils.segment_timeseries(x_train, window=self.forecast_window)

        full_pca = _pca()
        full_pca.fit(train_segments)

        cumulative_variance = np.cumsum(full_pca.explained_variance_ratio_)

        self.explained_variance = training_config["explained_variance"]
        self.num_components = np.argmax(cumulative_variance >= self.explained_variance) + 1
        self.cumulative_variance = cumulative_variance[self.num_components - 1]

        self.pca = _pca(n_components=self.num_components)
        self.pca.fit(train_segments)
        self.mean_ = self.pca.mean_

    def predict(self, x, reconstruct=False):
        if self.pca is None:
            raise ValueError("Model has not been initialized. Call fit() before predict().")

        segments = Utils.segment_timeseries(x, window=self.forecast_window)

        components = self.pca.transform(segments)
        pred = self.pca.inverse_transform(components)

        if reconstruct:
            pred, confint = Utils.average_reconstruct_timeseries_confint(pred)

        res = np.abs(x - pred)

        return pred, confint, res

    def save(self, save_path, filename):
        if self.pca is None:
            raise ValueError("There is no model to save. Please train the model first.")

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        with open(os.path.join(save_path, filename), 'wb') as file:
            pickle.dump(self, file)

        print(f"Model and configuration saved to {save_path}")

    @classmethod
    def load(cls, load_path, filename):
        with open(load_path+filename, 'rb') as file:
            model_instance = pickle.load(file)

        print(f"Model and configuration loaded from {load_path}")

        return model_instance
