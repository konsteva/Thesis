import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
from DataScraping.ArxivScrapper import ArxivScrapper

matplotlib.use('TKAgg')
font = {'size': 12}
matplotlib.rc('font', **font)

scraper = ArxivScrapper()

scraper.scrape("autoencoder")
scraper.save("autoencoder_publish_dates")

scraper.scrape(["autoencoder", "outlier detection"])
scraper.save("autoencoder_outlier_detection_publish_dates")

scraper.scrape(["autoencoder", "anomaly detection"])
scraper.save("autoencoder_anomaly_detection_publish_dates")

scraper.scrape(["autoencoder", "novelty detection"])
scraper.save("autoencoder_novelty_detection_publish_dates")

scraper.scrape(["anomaly detection"])
scraper.save("anomaly_detection_publish_dates")

scraper.scrape(["outlier detection"])
scraper.save("outlier_detection_publish_dates")

scraper.scrape(["novelty detection"])
scraper.save("novelty_detection_publish_dates")

autoencoder_data = pd.read_csv("autoencoder_publish_dates.csv", header=None)

anomaly_data = pd.read_csv("anomaly_detection_publish_dates.csv", header=None)
outlier_data = pd.read_csv("outlier_detection_publish_dates.csv", header=None)
novelty_data = pd.read_csv("novelty_detection_publish_dates.csv", header=None)

autoencoder_outlier_data = pd.read_csv("autoencoder_outlier_detection_publish_dates.csv", header=None)
autoencoder_anomaly_data = pd.read_csv("autoencoder_anomaly_detection_publish_dates.csv", header=None)
autoencoder_novelty_data = pd.read_csv("autoencoder_novelty_detection_publish_dates.csv", header=None)

anomaly_detection_data = pd.concat([anomaly_data, outlier_data, novelty_data])
anomaly_detection_data = anomaly_detection_data.reset_index()

autoencoder_anomaly_detection_data = pd.concat([autoencoder_outlier_data, autoencoder_anomaly_data, autoencoder_novelty_data])
autoencoder_anomaly_detection_data = autoencoder_anomaly_detection_data.reset_index()

# Dataframe
autoencoder_counts = pd.DataFrame(columns=["Year", "Count"])
autoencoder_counts["Year"] = np.unique(autoencoder_data[0].values, return_counts=True)[0]
autoencoder_counts["Count"] = np.unique(autoencoder_data[0].values, return_counts=True)[1]

anomaly_detection_counts = pd.DataFrame(columns=["Year", "Count"])
anomaly_detection_counts["Year"] = np.unique(anomaly_detection_data[0].values, return_counts=True)[0]
anomaly_detection_counts["Count"] = np.unique(anomaly_detection_data[0].values, return_counts=True)[1]

autoencoder_anomaly_detection_counts = pd.DataFrame(columns=["Year", "Count"])
autoencoder_anomaly_detection_counts["Year"] = np.unique(autoencoder_anomaly_detection_data[0].values, return_counts=True)[0]
autoencoder_anomaly_detection_counts["Count"] = np.unique(autoencoder_anomaly_detection_data[0].values, return_counts=True)[1]

merged_count = pd.merge(autoencoder_counts, anomaly_detection_counts,
                     on='Year', how='outer', suffixes=('_autoencoder', '_anomaly_detection'))
merged_count.fillna(0, inplace=True)

merged_count = pd.merge(merged_count, autoencoder_anomaly_detection_counts,
                        on='Year', how='outer')
merged_count.fillna(0, inplace=True)
merged_count.rename(columns={'Count': 'Count_autoencoder_anomaly_detection'}, inplace=True)
merged_count = merged_count.sort_values(by='Year')

fig, ax = plt.subplots(3, 1, figsize=(11, 6), sharex=True)

ax[0].plot(merged_count["Year"].values[1:], merged_count["Count_autoencoder"].values[1:], marker='o')
ax[2].set_xticks(list(range(min(merged_count["Year"].values[1:]), max(merged_count["Year"].values[1:]) + 1, 2)), rotation=45)
# ax[2].set_xlim(min(autoencoder_anomaly_detection_counts[0]) - 1, max(autoencoder_anomaly_detection_counts[0]) + 1)
ax[0].set_title("Autoencoder")

ax[1].plot(merged_count["Year"].values[1:], merged_count["Count_anomaly_detection"].values[1:], marker='o')
ax[2].set_xticks(list(range(min(merged_count["Year"].values[1:]), max(merged_count["Year"].values[1:]) + 1, 2)), rotation=45)
# ax[1].set_xlim(min(anomaly_detection_counts[0]) - 1, max(anomaly_detection_counts[0]) + 1)
ax[1].set_title("Anomaly Detection")

ax[2].plot(merged_count["Year"].values[1:], merged_count["Count_autoencoder_anomaly_detection"].values[1:], marker='o')
ax[2].set_xticks(list(range(min(merged_count["Year"].values[1:]), max(merged_count["Year"].values[1:]) + 1, 2)), rotation=45)
# ax[2].set_xlim(min(autoencoder_anomaly_detection_counts[0]) - 1, max(autoencoder_anomaly_detection_counts[0]) + 1)
ax[2].set_title("Autoencoder, Anomaly Detection")

# ax[0].bar(x=merged_count["Year"].values[10:], height=merged_count["Count_autoencoder"].values[10:])
# ax[2].set_xticks(list(range(min(merged_count["Year"].values[10:]), max(merged_count["Year"].values[10:]) + 1, 2)), rotation=45)
# # ax[2].set_xlim(min(autoencoder_anomaly_detection_counts[0]) - 1, max(autoencoder_anomaly_detection_counts[0]) + 1)
# ax[0].set_title("Autoencoder")
#
# ax[1].bar(x=merged_count["Year"].values[10:], height=merged_count["Count_anomaly_detection"].values[10:])
# ax[2].set_xticks(list(range(min(merged_count["Year"].values[10:]), max(merged_count["Year"].values[10:]) + 1, 2)), rotation=45)
# # ax[1].set_xlim(min(anomaly_detection_counts[0]) - 1, max(anomaly_detection_counts[0]) + 1)
# ax[1].set_title("Anomaly Detection")
#
# ax[2].bar(x=merged_count["Year"].values[10:], height=merged_count["Count_autoencoder_anomaly_detection"].values[10:])
# ax[2].set_xticks(list(range(min(merged_count["Year"].values[10:]), max(merged_count["Year"].values[10:]) + 1, 2)), rotation=45)
# # ax[2].set_xlim(min(autoencoder_anomaly_detection_counts[0]) - 1, max(autoencoder_anomaly_detection_counts[0]) + 1)
# ax[2].set_title("Autoencoder, Anomaly Detection")

plt.suptitle("Number of ArXiv papers containing keywords")
plt.tight_layout()
plt.show()
