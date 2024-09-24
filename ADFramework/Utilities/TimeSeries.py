import numpy as np
import pandas as pd

from ADFramework.Utilities import Utils


class TimeSeries:
    """ Helper class for univariate time-series """

    def __init__(self, ts_df=None, values=None, anomalies=None, anomaly_type=None, period=None, name=None):
        if ts_df is None and values is None and anomalies is None:
            raise Exception("Give either a dataframe of the time-series or a value-anomalies pair")

        if ts_df is None:
            if len(values) != len(anomalies):
                raise ValueError("Length of 'values' must be the same as the length of 'anomalies'")
            ts_df = pd.DataFrame(list(zip(values, anomalies)))

        if name is None:
            raise Exception("Name cannot be None as it must contain info on the timeseries")

        self.data = ts_df
        self.values = ts_df.iloc[:, 0].values
        self.anomalies = ts_df.iloc[:, 1].values
        self.name = name.split(".")[0]
        self.anomaly_type = anomaly_type
        self.anomaly_start = None if len(np.where(self.anomalies == 1)[0]) == 0 else np.where(self.anomalies == 1)[0][0]
        self.anomaly_end = None if len(np.where(self.anomalies == 1)[0]) == 0 else np.where(self.anomalies == 1)[0][-1]
        self.anomalous_points = None if self.anomaly_start is None else self.anomaly_end - self.anomaly_start + 1

        self.normal_points = len(np.where(self.anomalies == 0)[0])
        self.period = period
        self.test_start = int(self.name.split("_")[-3]) - 1
        self.train_values = self.values[:self.test_start]
        self.test_values = self.values[self.test_start:]
        self.array = self.array()

    def __len__(self):
        return len(self.values)

    def __str__(self):
        return (
            f"TimeSeries(Name={self.name}, Length={len(self)}, Period={self.period})")

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.values[index]
        else:
            return TimeSeries(values=self.values[index], anomalies=self.anomalies[index], name=self.name,
                              anomaly_type=self.anomaly_type, period=self.period)

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index < len(self.values):
            value = self.values[self._index]
            self._index += 1
            return value
        else:
            # When there are no more elements, raise StopIteration
            raise StopIteration

    def __repr__(self):
        return (
            f"TimeSeries(Name={self.name}, Length={len(self)}")

    def info(self):
        return {"Name": self.name,
                "Length": len(self),
                "No. Anomalous Points": self.anomalous_points,
                "Anomaly Type": self.anomaly_type,
                "Period": self.period
                }

    def array(self):
        return np.asarray(self.values), np.asarray(self.anomalies)

    def get_df(self):
        return self.data
