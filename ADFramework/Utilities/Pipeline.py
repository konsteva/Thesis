import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, r2_score, mean_squared_error, accuracy_score, precision_score, \
    recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

from ADFramework.Utilities import Utils
from ADFramework.Utilities.Scalers import Normalizer


class Pipeline:
    """
    Pipeline:
        1. Split
        2. Normalize
        3. Train
        4. Predict
        5. Anomaly detection
        6. Metrics
        7. Save metrics and plot
    """

    def __init__(self, train_val_ratio, normalizer, model, anomaly_score="residuals"):
        self.train_val_ratio = train_val_ratio
        self.normalizer = normalizer
        self.model = model
        if anomaly_score not in ["residuals", "confidence"]:
            raise Exception("Unknown anomaly score. Select either 'residuals', 'confidence'")
        self.anomaly_score = anomaly_score

    def apply(self, timeseries, train_config=None, early_stopping=None, save_dir=None, show_plot=False):
        try:
            period = int(timeseries.period)

            # ==================== 1. Splitting ==================== #
            # val_start = int(self.train_val_ratio * len(timeseries.train_values))
            #
            # train_vals = timeseries.train_values[:val_start]
            # val_vals = timeseries.train_values[val_start:]
            # test_vals = timeseries.test_values
            # train_anomalies = timeseries.anomalies[:val_start]
            # val_anomalies = timeseries.anomalies[val_start: timeseries.test_start]
            # test_anomalies = timeseries.anomalies[timeseries.test_start:]

            train_ts, val_ts, test_ts = Utils.split(timeseries, self.train_val_ratio)
            train_vals = train_ts.values
            val_vals = val_ts.values
            test_vals = test_ts.values
            test_anomalies = test_ts.anomalies

            # ==================== 2. Normalization ==================== #
            if self.normalizer:
                train_vals_norm = self.normalizer.fit_transform(train_vals)
                val_vals_norm = self.normalizer.transform(val_vals)
                test_vals_norm = self.normalizer.transform(test_vals)
            else:
                train_vals_norm = train_vals
                val_vals_norm = val_vals
                test_vals_norm = test_vals

            # ===================== 3. Fitting ===================== #
            history = self.model.fit(train_vals_norm, val_vals_norm, train_config, early_stopping)

            # ==================== 4. Forecasts ==================== #
            train_forecasts, train_conf_ints, train_residuals = self.model.predict(train_vals_norm, reconstruct=True)
            val_forecasts, val_conf_ints, val_residuals = self.model.predict(val_vals_norm, reconstruct=True)
            test_forecasts, test_conf_ints, anomaly_scores = self.model.predict(test_vals_norm, reconstruct=True)

            if self.anomaly_score == "confidence":
                lower_bound = test_conf_ints[:, 0]
                upper_bound = test_conf_ints[:, 1]

                confidence = upper_bound - lower_bound
                anomaly_scores = confidence

            # ==================== 5. Anomaly detection ==================== #
            # Calculate AUC for residuals
            res_normalizer = Normalizer()
            residuals_prob = res_normalizer.fit_transform(np.nan_to_num(anomaly_scores))

            # Calculate optimal residual threshold
            res_auc = roc_auc_score(test_anomalies,  np.nan_to_num(residuals_prob))
            fpr, tpr, res_thresholds = roc_curve(test_anomalies,  np.nan_to_num(residuals_prob))
            res_youdens_j = tpr - fpr
            res_optimal_threshold = res_thresholds[np.argmax(res_youdens_j)]
            res_optimal_threshold = res_normalizer.inverse(res_optimal_threshold)

            # Calculate AUC for windowed residuals
            window_anomaly_scores = np.convolve(np.abs(anomaly_scores), np.ones(period, dtype=int), 'valid')

            pad_test = len(test_vals) - len(window_anomaly_scores)
            window_anomaly_scores = Utils.start_pad_end_pad(window_anomaly_scores, pad_test, padding_value=np.nan)

            # Calculate optimal windowed residual threshold
            window_res_normalizer = Normalizer()
            window_res_prob = window_res_normalizer.fit_transform(np.nan_to_num(window_anomaly_scores))

            window_res_auc = roc_auc_score(test_anomalies, np.nan_to_num(residuals_prob))
            fpr, tpr, thresholds = roc_curve(test_anomalies, np.nan_to_num(window_res_prob))
            window_youdens_j = tpr - fpr
            window_res_optimal_threshold = thresholds[np.argmax(window_youdens_j)]
            window_res_optimal_threshold = window_res_normalizer.inverse(window_res_optimal_threshold)

            # Predict anomalies
            # Use best (highest ROC-AUC score) optimal window threshold to classify anomalies
            threshold = res_optimal_threshold if res_auc > window_res_auc else window_res_optimal_threshold

            forecasted_anomalies = np.zeros(len(test_vals))
            forecasted_anomalies[np.where(window_anomaly_scores > threshold)[0]] = 1

            # Fill in-between close anomalies
            forecasted_anomalies = Utils.fill_anomalies(forecasted_anomalies, threshold=period)
            pred_anomalies_idx = [[idx for idx in anomaly] for anomaly in Utils.get_distinct_anomalies(forecasted_anomalies)]
            pred_anomalies_idx_adj = [[idx + timeseries.test_start for idx in anomaly] for anomaly in Utils.get_distinct_anomalies(forecasted_anomalies)]

            # Cumulative residual per predicted anomaly
            cum_res = [np.sum(window_anomaly_scores[pred_anomalies_idx[i]]) for i in range(len(pred_anomalies_idx))]

            # Most notable anomaly
            notable_anomaly_idx = np.argmax(cum_res)
            pred_anomalies_idx = pred_anomalies_idx[notable_anomaly_idx]
            pred_anomalies_idx_adj = pred_anomalies_idx_adj[notable_anomaly_idx]
            forecasted_anomalies = np.zeros(len(test_vals), dtype=int)
            forecasted_anomalies[pred_anomalies_idx] = 1

            # ==================== 6. Calculating metrics ==================== #
            # print("6. Calculating metrics")
            # Regression
            train_true = train_vals_norm
            val_true = val_vals_norm
            test_true = test_vals_norm
            train_pred = train_forecasts
            val_pred = val_forecasts
            test_pred = test_forecasts
            true_labels = timeseries.anomalies[timeseries.test_start:]
            pred_labels = forecasted_anomalies

            # find number of nan values
            idx = 0
            while np.isnan(train_pred[idx]):
                idx += 1

            train_true = train_true[idx:]
            train_pred = train_pred[idx:]
            val_true = val_true[idx:]
            val_pred = val_pred[idx:]
            test_true = test_true[idx:]
            test_pred = test_pred[idx:]

            train_r2 = r2_score(train_true, train_pred)
            val_r2 = r2_score(val_true, val_pred)
            test_r2 = r2_score(test_true, test_pred)

            train_mse = mean_squared_error(train_true, train_pred)
            val_mse = mean_squared_error(val_true, val_pred)
            test_mse = mean_squared_error(test_true, test_pred)

            train_smape = Utils.smape(train_true, train_pred)
            val_smape = Utils.smape(val_true, val_pred)
            test_smape = Utils.smape(test_true, test_pred)

            # Classification
            accuracy = accuracy_score(true_labels, pred_labels)
            precision = precision_score(true_labels, pred_labels)
            recall = recall_score(true_labels, pred_labels)
            f1 = f1_score(true_labels, pred_labels)

            minmax = Normalizer()
            window_anomaly_scores_prob = minmax.fit_transform(np.nan_to_num(window_anomaly_scores, copy=True, nan=0.0))
            auc = roc_auc_score(true_labels, window_anomaly_scores_prob)

            # Anomaly detection
            anomaly_found = precision > 0.25

            cm = confusion_matrix(true_labels, pred_labels)

            # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            #
            # disp.plot()

            FP = cm.sum(axis=0) - np.diag(cm)
            FN = cm.sum(axis=1) - np.diag(cm)
            TP = np.diag(cm)
            TN = cm.sum() - (FP + FN + TP)

            FPR = FP/(FP+TN)
            FNR = FN/(TP+FN)

            # ==================== 7. Save metrics and prediction plot ========== #
            # Metrics
            metrics = pd.DataFrame([[timeseries.name, train_r2, val_r2, test_r2, train_mse, val_mse, test_mse,
                                     train_smape, val_smape, test_smape, accuracy, precision, recall,
                                     f1, auc, anomaly_found, FPR[1], FNR[1]]],
                                   columns=["Timeseries", "Training R2", "Validation R2", "Test R2", "Training MSE", "Validation MSE", "Test MSE",
                                            "Training SMAPE", "Validation SMAPE", "Test SMAPE", "Accuracy", "Precision", "Recall",
                                            "F1", "AUC", "Found", "FPR", "FNR"])

            # Fix data for plotting
            true_x = list(range(timeseries.test_start + idx, len(timeseries)))
            true_y = timeseries.test_values[idx:]

            pred_x = true_x
            if self.normalizer:
                pred_y = self.normalizer.inverse(test_pred)
                pred_conf_int_low = self.normalizer.inverse(test_conf_ints[:, 0])
                pred_conf_int_up = self.normalizer.inverse(test_conf_ints[:, 1])
            else:
                pred_y = test_pred
                pred_conf_int_low = test_conf_ints[:, 0][idx:]
                pred_conf_int_up = test_conf_ints[:, 1][idx:]

            true_anomaly_start = timeseries.anomaly_start
            true_anomaly_end = timeseries.anomaly_end

            pred_anomalies = [pred_anomalies_idx_adj]
            pred_conf_ints = np.column_stack((pred_conf_int_low, pred_conf_int_up))

            # ====================== Time Series & Predictions plots  ====================== #
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 6))

            # True values
            true_line, = ax1.plot(true_x, true_y, label="Ground truth")

            # Forecasted values
            forecast_line, = ax1.plot(pred_x, pred_y, label="Forecasts")

            # Forecast conf int
            conf_int_fill = ax1.fill_between(pred_x, pred_conf_ints[:, 0], pred_conf_ints[:, 1],
                                             color='lightblue', alpha=0.5, label="5% Confidence Interval")

            # True anomalies
            y_min, y_max = ax1.get_ylim()
            true_anom_rect = ax1.axvspan(true_anomaly_start, true_anomaly_end, color="red", alpha=0.2, label="True Anomalies")

            # Pred anomalies
            pred_anom_rects = []
            for i, anomaly in enumerate(pred_anomalies):
                if i == 0:
                    rect = ax1.axvspan(anomaly[0], anomaly[-1], color="green", alpha=0.2, label="Predicted Anomalies")
                    pred_anom_rects.append(rect)
                else:
                    rect = ax1.axvspan(anomaly[0], anomaly[-1], color="green", alpha=0.2)

            # Add ax1 specific legend
            ax1.legend(handles=[true_line, forecast_line], loc="upper left")

            # Add the general plot legend excluding "Ground truth" and "Forecasts"
            fig.legend(loc="upper right", handles=[conf_int_fill, true_anom_rect] + pred_anom_rects)

            # ====================== Simple Residuals & Threshold  ====================== #
            residuals_line, = ax2.plot(true_x, anomaly_scores[idx:], label="Confidence" if self.anomaly_score == "confidence" else "Residuals")
            threshold_line = ax2.axhline(res_optimal_threshold, linewidth=0.8, linestyle="--", color="red", label="Threshold")

            # True anomalies
            ax2.axvspan(true_anomaly_start, true_anomaly_end, color="red", alpha=0.2)

            # Pred anomalies
            for anomaly in pred_anomalies:
                ax2.axvspan(anomaly[0], anomaly[-1], color="green", alpha=0.2)

            # Add ax2 specific legend
            ax2.legend(loc="upper left")

            # ====================== Windowed Residuals & Threshold  ====================== #
            window_residuals_line, = ax3.plot(true_x, window_anomaly_scores[idx:], label="Windowed Confidence" if self.anomaly_score == "confidence" else "Windowed Residuals")
            window_threshold_line = ax3.axhline(window_res_optimal_threshold, linewidth=0.8,
                                                linestyle="--", color="red", label="Threshold")

            # True anomalies
            ax3.axvspan(true_anomaly_start, true_anomaly_end, color="red", alpha=0.2)

            # Pred anomalies
            for anomaly in pred_anomalies:
                ax3.axvspan(anomaly[0], anomaly[-1], color="green", alpha=0.2)

            # Add ax3 specific legend
            ax3.legend(loc="upper left")

            plt.suptitle(f"{timeseries.name}")
            plt.tight_layout()

            if show_plot:
                plt.show()

            # ====================== Save metrics, plot and model ====================== #
            if save_dir is not None:
                metrics.to_csv(save_dir + timeseries.name + "_metrics.csv", index=False)
                pickle.dump(fig, open(save_dir + f"{timeseries.name}_plot.fig.pickle", 'wb'))
                self.model.save(save_dir, timeseries.name + "_autoencoder.zip")

        except Exception as e:
            message = f"The following excption occured for timeseries {timeseries.name}:\n{e}"
            print("\033[41m{}\033[0m".format(message))
