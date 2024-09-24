import numpy as np
from ADFramework.Utilities.Utils import TimeSeries
from abc import ABC, abstractmethod


class Scaler(ABC):
    @abstractmethod
    def fit(self, x):
        pass

    @abstractmethod
    def transform(self, x):
        pass

    @abstractmethod
    def fit_transform(self, x):
        pass

    @abstractmethod
    def inverse(self, x):
        pass


class Normalizer(Scaler):
    """ MinMax Scaler """

    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, timeseries):
        if isinstance(timeseries, TimeSeries):
            X = timeseries.values
        else:
            X = timeseries
        self.min = np.min(X)
        self.max = np.max(X)

        return self

    def transform(self, timeseries):
        if self.min is None or self.max is None:
            raise Exception("Normalizer has not been initialized yet. Call fit method first")

        if isinstance(timeseries, TimeSeries):
            X = timeseries.values
        else:
            X = timeseries
        X_norm = (X - self.min) / (self.max - self.min)

        return X_norm

    def fit_transform(self, timeseries):
        self.fit(timeseries)
        return self.transform(timeseries)

    def inverse(self, norm_values):
        return (self.max - self.min) * norm_values + self.min


class Standardizer(Scaler):
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, timeseries):
        if isinstance(timeseries, TimeSeries):
            X = timeseries.values
        else:
            X = timeseries
        self.mean = np.mean(X)
        self.std = np.std(X)

        return self

    def transform(self, timeseries):
        if self.mean is None or self.std is None:
            raise Exception("Standardizer has not been initiallized yet. Call fit method first")

        if isinstance(timeseries, TimeSeries):
            X = timeseries.values
        else:
            X = timeseries

        timeseries_norm = (X - self.mean) / self.std

        return timeseries_norm

    def fit_transform(self, timeseries):
        """ Can be used for a 'per-segment' normalization where the timeseries input variable is a segment """
        self.fit(timeseries)
        return self.transform(timeseries)

    def inverse(self, timeseries_norm):
        return self.std * timeseries_norm + self.mean
