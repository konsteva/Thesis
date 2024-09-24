import os

import numpy as np
from statsmodels.iolib import load_pickle, save_pickle
from tqdm import tqdm
from statsmodels.tsa.statespace.sarimax import SARIMAX

from ADFramework.Models.Model import Model
from ADFramework.Utilities import Utils


class SARIMA(Model):
    def __init__(self, order=None, seasonal_order=None, forecast_window=None):
        super().__init__("SARIMA")
        if order is None:
            self.params = [1, 0, 0]

        if seasonal_order is None:
            self.seasonal_params = [1, 0, 0]

        self.is_compiled = False
        self.order = order
        self.seasonal_order = seasonal_order
        self.forecast_window = forecast_window
        self.sarima = None

    def fit(self, x_train, x_val, training_config=None, early_stopping=None):
        if not self.is_compiled:
            self.initialize(x_train)
            self.is_compiled = True
        self.sarima = self.sarima.fit()

    def predict(self, x, reconstruct=True):
        """ Performs a rolling forecast on the input array """
        if not self.is_compiled:
            raise Exception("Make sure the model is fitted before predicting")

        forecasts = []
        conf_ints = []
        if self.forecast_window is None:
            self.forecast_window = Utils.find_period(x)

        for t in tqdm(range(0, len(x), self.forecast_window)):
            forecast = self.sarima.get_forecast(int(self.forecast_window))
            forecasts.extend(forecast.predicted_mean)  # use train values for the first forecasts
            conf_ints.extend(forecast.conf_int(alpha=0.05))
            self.sarima = self.sarima.append(x[t: t + self.forecast_window + 1], refit=False)  # add the ground truth test values for next forecasts

        forecasts = np.asarray(forecasts)
        conf_ints = np.asarray(conf_ints)

        # Fix array size -> order = 1 so predictions start after first value
        if len(forecasts) >= len(x):
            forecasts = forecasts[:len(x) - 1]
            conf_ints = conf_ints[:len(x) - 1]
            forecasts = np.insert(forecasts, 0, np.nan)
            conf_ints = np.insert(conf_ints, 0, np.nan, axis=0)

        residuals = np.abs(x - forecasts)

        return forecasts, conf_ints, residuals

    def initialize(self, x_train):
        self.sarima = SARIMAX(x_train, order=self.order, seasonal_order=self.seasonal_order)

    def save(self, save_dir, file_name):
        if ".pickle" not in file_name:
            file_name += ".pickle"

        save_pickle(self.sarima, os.path.join(save_dir, file_name))

    @classmethod
    def load(cls, model_dir, file_name):
        loaded_sarima = load_pickle(os.path.join(model_dir, file_name))
        ar = loaded_sarima.model_orders["ar"]
        diff = loaded_sarima.model_orders["diff"]
        ma = loaded_sarima.model_orders["ma"]

        s_ar = loaded_sarima.seasonal_periods["ar"]
        s_diff = loaded_sarima.seasonal_periods["diff"]
        s_ma = loaded_sarima.seasonal_periods["ma"]

        period = loaded_sarima.seasonal_periods["seasonal_periods"]

        sarima = cls(order=[ar, diff, ma], seasonal_order=[s_ar, s_diff, s_ma], forecast_window=period)
        sarima.is_compiled = True

        return sarima
