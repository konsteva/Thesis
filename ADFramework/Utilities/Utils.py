import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from scipy.stats import iqr
# from statsmodels.tsa._stl import STL
from statsmodels.tsa.stattools import acf, adfuller
from tabulate import tabulate
import matplotlib.transforms as mtransforms
from tqdm import tqdm
import os

from ADFramework.Utilities.TimeSeries import TimeSeries

matplotlib.use('TKAgg')


def load_timeseries(ts_path, metadata_path=None):
    ts = pd.read_csv(ts_path)

    if "\\" in ts_path:
        splitter = "\\"
    else:
        splitter = "/"

    name = ts_path.split(splitter)[-1]
    idx = int(name.split("_")[0])
    period = None
    anomaly_type = None
    if metadata_path is not None:
        metadata = pd.read_csv(metadata_path)
        period = metadata[metadata["TimeSeries #"] == idx]["Period"].values[0]
        anomaly_type = metadata[metadata["TimeSeries #"] == idx]["Anomaly Type"].values

    return TimeSeries(ts, anomaly_type=anomaly_type, period=period, name=name)


def downsample_timeseries(timeseries, downsample=2):
    """
    Downsamples a time series by windowing and averaging each window
    :param timeseries: the time series to perform downsampling
    :param downsample: the downsampling factor aka the windowing size
    :return: TimeSeries object of the downsampled timeseries
    """
    data = timeseries.data

    anomalies_col = "Anomalies" if "Anomalies" in data.columns else 1
    data["group"] = np.floor(np.arange(len(data)) / downsample)

    resampled_data = data.groupby("group").mean().reset_index(drop=True)
    resampled_data.loc[resampled_data[anomalies_col] > 0, anomalies_col] = 1

    if len(resampled_data[resampled_data[anomalies_col] == 1]) == 0:
        raise Exception("No anomalies in the downsampled time series")

    name_info = timeseries.name.split("_")
    name_info[-3:] = [str(int(num)//downsample) for num in name_info[-3:]]
    new_name = "_".join(name_info)

    return TimeSeries(
        resampled_data,
        anomaly_type=timeseries.anomaly_type,
        period=timeseries.period//downsample,
        name=new_name)


def load_UCR(data_dir, metadata_dir=None):
    all_ts = []
    for filename in tqdm(os.listdir(data_dir)):
        if (os.path.isdir(os.path.join(data_dir, filename)) or
                "distorted" in filename.lower() or "noise" in filename.lower() or
                not filename.endswith('.csv')
        ):
            continue

        file_path = os.path.join(data_dir, filename)
        ts = load_timeseries(file_path, metadata_dir)

        all_ts.append(ts)

    return all_ts


def find_period(timeseries, return_all=False):
    """ Highest peak in the acf plot indicates the period. The peak points are located where the second order
    differences become negative. Of those, the index of the highest is kept as the period """

    if isinstance(timeseries, TimeSeries):
        timeseries = timeseries.values

    acf_vals = acf(timeseries, nlags=min(10000, len(timeseries)))
    inflection = np.diff(np.sign(np.diff(acf_vals)))
    peaks = (inflection < 0).nonzero()[0] + 1
    period = peaks[acf_vals[peaks].argmax()]

    if return_all:
        return int(period), acf_vals, peaks

    return int(period)


def segment_timeseries(timeseries, window="auto"):
    stride = 1  # hardcoded for now
    sampling_rate = 1  # hardcoded for now

    if window == "auto":
        window = find_period(timeseries)

    segments = []
    total_segments = max(0, int((len(timeseries) - window) / stride) + 1)
    for i in tqdm(range(total_segments)):
        rnd = np.random.random()
        if rnd < sampling_rate:
            segment = timeseries[i: i + window]
            segments.append(segment)

    return np.array(segments)


def split(timeseries, train_val_ratio=0.7):
    val_start = int(train_val_ratio * (timeseries.test_start - 1))
    train_ts = timeseries[:val_start]
    val_ts = timeseries[val_start: timeseries.test_start]
    test_ts = timeseries[timeseries.test_start:]

    return train_ts, val_ts, test_ts


def ts_idx_to_segment(window, stride, idx):
    """ Finds the segment index that ends in the item with index: idx. E.g.:
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] -> w=3, s=1
        [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10]
        idx = 5 -> [3, 4, 5] -> seg_idx=3

        Formula: idx - (window - s) """
    return idx - (window - stride)


def segment_idx_to_ts_idx(idx, window):
    stride = 1  # hardcoded for now
    start_index = stride * idx
    end_index = stride * idx + window - 1

    return start_index, end_index


def calculate_statistics(array, return_table=False):
    minimum = np.min(array)
    maximum = np.max(array)
    mean = np.mean(array)
    median = np.median(array)
    std = np.std(array)
    q1 = np.quantile(array, 0.25)
    q3 = np.quantile(array, 0.75)

    if return_table:
        table = [["Statistics", "Values"], ["Minimum", minimum], ["Maximum", maximum], ["Mean", mean],
                 ["Median", median], ["STD", std], ["Q1", q1], ["Q3", q3]]
        return tabulate(table, tablefmt="grid")

    return {"min": minimum, "max": maximum, "mean": mean, "median": median, "std": std, "q1": q1, "q3": q3,
            "iqr": q3 - q1}


def simple_reconstruct_timeseries(segments):
    """
    If a time series is split into segments it reconstructs it back to timeseries
    :param segments: 2d ndarray of shape (n_segments, window_size)
    :return: ndarray of the reconstructed timeseries values
    """
    stride = 1  # hardcoded for now
    ts = []
    for segment in segments[:-1]:
        ts.extend(segment[:stride].tolist())

    ts.extend(segments[-1])

    return np.asarray(ts)


def average_reconstruct_timeseries_confint(segments):
    """
    Given a ndarray of segments it reconstructs a time series by averaging overlapping values
    :param segments: segments: 2d ndarray of shape (n_segments, window_size)
    :return: ndarray of the reconstructed averaged timeseries values
    """
    stride = 1  # hardcoded for now
    window = segments.shape[1]
    ts_len = len(segments) + segments.shape[1] - 1  # n = floor((N - w) / 1) + 2
    average = np.zeros(ts_len)
    minimum = np.zeros(ts_len)
    maximum = np.zeros(ts_len)
    counter = np.zeros(ts_len)

    minimum[0: window] = segments[0]
    maximum[0: window] = segments[0]

    for i, pred_segment in enumerate(segments):
        # average
        average[i * stride: len(pred_segment) + i * stride] += pred_segment
        counter[i * stride: len(pred_segment) + i * stride] += 1

        if i == 0:
            continue

        # minimum
        previous_min_seg = minimum[i * stride: len(pred_segment) + i * stride - 1]  # ignore last element since it will be added later -> line 210
        minimum[i * stride: len(pred_segment) + i * stride - 1] = np.minimum(previous_min_seg, pred_segment[:-1])
        minimum[i * stride + len(pred_segment) - 1] = pred_segment[-1]

        # maximum
        previous_max_seg = maximum[i * stride: len(pred_segment) + i * stride - 1]  # ignore last element since it will be added later -> line 210
        maximum[i * stride: len(pred_segment) + i * stride - 1] = np.maximum(previous_max_seg, pred_segment[:-1])
        maximum[i * stride + len(pred_segment) - 1] = pred_segment[-1]

    average = average / counter

    conf_int = np.column_stack((minimum, maximum))

    return average, conf_int


def count_distinct_anomalies(input_vals, threshold=1):
    """
    Counts number of discrete anomalies of a timeseries. Includes both point and sequential anomalies. Threshold is
    the maximum distance between two anomalous indices to be considered as the same anomaly.
    :param input_vals: Timeseries or ndarray. If Timeseries then anomalous indices are extracted. If ndarray anomalies are directly calculated
    :param threshold: The distance between two anomalous points to be considered part of the same anomaly
    :return: (int) number of distinct anomalies
    """
    if isinstance(input_vals, TimeSeries):
        anomalous_points = np.where(input_vals.anomalies == 1)[0]
    else:
        anomalous_points = input_vals

    if anomalous_points.size == 0:
        return 0
    num_anomalies = min(1, len(anomalous_points))
    current_idx = anomalous_points[0]
    for idx in anomalous_points[1:]:
        if idx > current_idx + 1 + threshold:
            num_anomalies += 1

        current_idx = idx

    return num_anomalies


def get_distinct_anomalies(ts_or_anomalies_arr):
    """
    Returns the indices of each distinct anomaly
    :param ts_or_anomalies_arr: timeseries or numpy array
    :return:
    """
    if isinstance(ts_or_anomalies_arr, TimeSeries):
        anomalies = ts_or_anomalies_arr.anomalies
    else:
        anomalies = ts_or_anomalies_arr

    anomalies_idx = np.where(anomalies == 1)[0]
    if len(anomalies_idx) == 0:
        return []

    distinct_anomalies_idx = []
    current_anomaly_idx = [anomalies_idx[0]]
    for i in range(1, len(anomalies_idx)):
        if (anomalies_idx[i] == anomalies_idx[i - 1] + 1):
            current_anomaly_idx.append(anomalies_idx[i])
        else:
            distinct_anomalies_idx.append(current_anomaly_idx)
            current_anomaly_idx = [anomalies_idx[i]]

    distinct_anomalies_idx.append(current_anomaly_idx)

    return distinct_anomalies_idx


def find_point_anomalies(timeseries):
    """
    Returns the indices of point anomalies in a time series
    """
    indices = np.where(timeseries.anomalies == 1)[0]
    point_anomalies_indices = []

    for i in range(len(indices)):
        current_idx = indices[i]
        if i == 0 or current_idx != indices[i - 1] + 1:
            # Check if the anomaly has length 1
            if i == len(indices) - 1 or current_idx + 1 != indices[i + 1]:
                point_anomalies_indices.append(current_idx)

    return point_anomalies_indices


def find_sequential_anomalies(timeseries):
    indices = np.where(timeseries.anomalies == 1)[0]
    point_anomalies_indices = find_point_anomalies(timeseries)
    seq_idx = [idx for idx in indices if idx not in point_anomalies_indices]
    sequential_anomalies_indices = []
    i = 0
    while i < len(seq_idx):
        temp = [seq_idx[i]]
        while i < len(seq_idx) - 1 and seq_idx[i] == seq_idx[i + 1] - 1:
            temp.append(seq_idx[i + 1])
            i += 1

        sequential_anomalies_indices.append(temp)
        i += 1

    return sequential_anomalies_indices


def anomaly_found(anomaly_start, anomaly_end, predicted_anomalies):
    """
    An anomaly is considered found if any of the predicted points lies inside the range (anomaly_start, anomaly_end)
    :param anomaly_start: The starting index of the anomaly
    :param anomaly_end: The ending index of the anomaly
    :param predicted_anomalies: 1d list or ndarray of predicted anomalous indices
    :return: boolean - whether the actual anomaly was found or not
    """
    found = False
    for prediction in predicted_anomalies:
        if prediction in range(anomaly_start, anomaly_end + 1):
            found = True

    return found


def plot(timeseries, title=None, size=(10, 5), ax=None, anomaly_color="red"):
    """ Plots the time series along with its anomaly"""
    x = list(range(len(timeseries)))
    values = timeseries.values
    anomalies = timeseries.anomalies
    point_anomaly_indices = find_point_anomalies(timeseries)
    sequential_anomaly_indices = find_sequential_anomalies(timeseries)
    has_points_anomalies = len(point_anomaly_indices) != 0
    has_sequential_anomalies = len(sequential_anomaly_indices) != 0

    if ax is None:
        _, ax = plt.subplots(figsize=tuple(size))

    # plot signal
    ax.plot(x, values)

    # plot point anomalies
    if has_points_anomalies:
        for i, point in enumerate(point_anomaly_indices):
            if i == 0:
                ax.axvline(point, color=anomaly_color, alpha=0.2, label="Point Anomalies")
            else:
                ax.axvline(point, color=anomaly_color, alpha=0.2)

    where = np.zeros(len(anomalies))
    for seq_anomaly in sequential_anomaly_indices:
        for idx in seq_anomaly:
            where[idx] = 1

    y_min, y_max = plt.gca().get_ylim()
    # plot sequential anomalies
    if has_sequential_anomalies:
        # trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.fill_between(x, y_min, y_max, where=where == 1, facecolor=anomaly_color, alpha=0.2,
                        # transform=trans,
                        label="Sequential Anomalies")

    if has_points_anomalies or has_sequential_anomalies:
        plt.legend()

    if title is not None:
        plt.title(title)

    plt.suptitle(f"{timeseries.name}")
    plt.show()

    return ax


def is_stationary(timeseries, confidence=0.95):
    """ ADF test to determine stationarity of a time series """
    if isinstance(timeseries, TimeSeries):
        timeseries = timeseries.values

    # Perform the ADF test
    adf_result = adfuller(timeseries)
    p_value = adf_result[1]
    is_stationary = p_value < (1 - confidence)

    return is_stationary


def start_pad_end_pad(arr, total_padding, padding_value=0):
    """
    Adds an equal number of padding at beginning and end of an array
    :param arr: the ndarray to add padding to
    :param total_padding: the total padding to add (sum of beginning and end pad)
    :param padding_value: the value to pad (default is zero but could also be NaN or any other number)
    :return: the padded array
    """
    start_pad = (total_padding) // 2
    end_pad = (total_padding) - start_pad

    start_pad_arr = np.full(start_pad, padding_value)
    padded_arr = np.concatenate((start_pad_arr, arr))

    end_pad_arr = np.full(end_pad, padding_value)
    padded_arr = np.concatenate((padded_arr, end_pad_arr))

    return padded_arr


def fill_anomalies(input_vals, threshold=1):
    """
    If two anomalies are closer than the threshold then they are unified as one discrete anomaly instead of two by
    filling their distance with anomalies
    :param input_vals: the array of anomaly labels
    :param threshold: the threshold distance to unify two anomalies
    :return: the editted anomaly labels
    """
    if isinstance(input_vals, TimeSeries):
        anomalous_points = np.where(input_vals.anomalies == 1)[0]
    else:
        anomalous_points = input_vals

    if anomalous_points.size == 0:
        return 0

    anomalous_idx = np.where(anomalous_points == 1)[0]  # Find indices of ones

    for i in range(len(anomalous_idx) - 1):
        start = anomalous_idx[i]
        end = anomalous_idx[i + 1]

        # Check if the distance between two ones is less than or equal to the threshold
        if end - start - 1 <= threshold:
            anomalous_points[start:end + 1] = 1  # Fill the range between the indices with ones

    return anomalous_points


def smape(a, b):
    """ Calculates the Symmetric Mean Absolute Percentage Error (SMAPE) between two arrays """
    smapes = np.abs(np.subtract(a, b)) / (np.abs(a) + np.abs(b))
    mask = (a == 0) & (b == 0)

    smapes[mask] = np.nan

    return np.nanmean(smapes)


def results_metadata(metadata_path, results_path):
    # Load individual results and create dataframe with all
    results = pd.DataFrame()
    for filename in tqdm(os.listdir(results_path)):
        if not "metrics" in filename:
            continue

        file_path = os.path.join(results_path, filename)

        ts_results = pd.read_csv(file_path)
        results = pd.concat([results, ts_results])

    results = results.reset_index(drop=True)

    # Load and format metadata dataframe
    metadata = pd.read_csv(metadata_path, header=0)
    cols = ['Name', 'Training idx', 'Anomaly Start', 'Anomaly End']
    metadata['Timeseries'] = metadata[cols].apply(lambda row: 'UCR_Anomaly_'+'_'.join(row.values.astype(str)), axis=1)
    metadata.insert(0, "Timeseries", metadata.pop("Timeseries"))
    metadata['Timeseries'] = metadata[["TimeSeries #", "Timeseries"]].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

    # Merge metadata and results
    results = pd.merge(metadata, results, on="Timeseries")

    return results


def kolmogorov_smirnov(errors_matrix):
    """
    Performs Kolmogorov-Smirnov normality test on the predicitons for each time step
    :param errors_matrix: A nxm ndarray with all residuals for each prediction, for n segments and m residuals per segment (segment size)
    """
    def _per_timestep_errors(all_errors):
        # invert columns order
        all_errors = all_errors[:, ::-1]
        all_errors = np.array(all_errors)
        m, n = errors_matrix.shape

        diagonals = []

        # Extract all diagonals
        for d in range(-m + 1, n):
            diag = np.diagonal(all_errors, offset=d)
            diagonals.append(list(diag))

        return diagonals

    # Example array x
    errors = _per_timestep_errors(errors_matrix)

    # Perform the Kolmogorov-Smirnov test for normality
    non_normal = 0
    for error in errors:
        mean = np.mean(error)
        std = np.std(error)
        ks_statistic, p_value = kstest(error, 'norm', args=(mean, std))

        if p_value < 0.05:
            non_normal += 1

    print("% of normally distributed errors: ", non_normal)
    print("% of non-normally distributed errors: ", non_normal / len(errors))
