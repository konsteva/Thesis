import pandas as pd
import os
from tqdm import tqdm

directory = "Z:\\Eva\\Μεταπτυχιακό\\Thesis\\Data\\UCR_AnomalyDatasets_2021\\UCR_TimeSeriesAnomalyDatasets2021\\FilesAreInHere\\UCR_Anomaly_FullData"
save_dir = "Z:\\Eva\\Μεταπτυχιακό\\Thesis\\UCR_Dataset\\Edited UCR Dataset\\Mini UCR Dataset\\noise"
for file_name in tqdm(os.listdir(directory)):
    path = os.path.join(directory, file_name)
    info = file_name[:-4].split("_")
    name = file_name.split(".")[0]
    if os.path.isdir(path) or name.lower().startswith(("distorted", "noise")) or not file_name.endswith("txt"):
        continue

    train_end = int(info[4])
    anomaly_start = int(info[5])
    anomaly_end = int(info[6])
    # ignore folders and keep only original datasets

    file_path = os.path.join(directory, file_name)
    data = pd.read_csv(file_path, header=None)
    data[1] = [0] * len(data)
    data.iloc[anomaly_start:anomaly_end+1, 1] = [1] * (anomaly_end - anomaly_start + 1)
    # data[2] = ["Test"] * len(data)
    # data.iloc[:train_end, 2] = ["Training"] * train_end

    data.columns = ["Values", "Anomalies"]
    data.to_csv(os.path.join(save_dir, name+".csv"), index=False, header=True)
