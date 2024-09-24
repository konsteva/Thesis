from ADFramework.Models.Autoencoders import VanillaAutoencoder, FusionAutoencoder
from ADFramework.Models.Autoregressive import SARIMA
from ADFramework.Models.PCA import PCA
from ADFramework.Utilities import Utils
from ADFramework.Utilities.Pipeline import Pipeline
from ADFramework.Utilities.Scalers import Normalizer

data_dir = "Z:\\Eva\\Μεταπτυχιακό\\Thesis\\UCR_Dataset\\Edited UCR Dataset\\Extended Data"
metadata_dir = "Z:\\Eva\\Μεταπτυχιακό\\Thesis\\UCR_Dataset\\Edited UCR Dataset\\Metadata\\Metadata.csv"

if __name__ == "__main__":
    all_ts = Utils.load_UCR(data_dir, metadata_dir)
    ts = all_ts[4]

    period = ts.period

    # # PCA
    # model = PCA(period)
    # train_config = {"explained_variance": 0.9}
    # normalizer = Normalizer()
    # pipeline = Pipeline(train_val_ratio=0.7, normalizer=normalizer, model=model, anomaly_score="residuals")
    # pipeline.apply(ts, train_config, early_stopping=None, save_dir="./", show_plot=True)

    # # SARIMA
    # model = SARIMA(order=[1, 0, 1], seasonal_order=[1, 0, 1, period], forecast_window=period)
    # train_config = None
    # normalizer = None
    # pipeline = Pipeline(train_val_ratio=0.7, normalizer=normalizer, model=model, anomaly_score="residuals")
    # pipeline.apply(ts, train_config, early_stopping=None, save_dir="./", show_plot=True)

    # # Vanilla Autoencoder
    # model = VanillaAutoencoder(input_size=period)
    # training_config = {"optimizer": "adam", "loss": "mse", "epochs": 10, "batch_size": 32}
    # normalizer = Normalizer()
    # pipeline = Pipeline(train_val_ratio=0.7, normalizer=normalizer, model=model, anomaly_score="residuals")
    # pipeline.apply(ts, training_config, early_stopping=None, save_dir="./", show_plot=True)

    # Fusion Autoencoder
    model = FusionAutoencoder(input_size=period)
    training_config = {"autoencoder": {"optimizer": "adam", "loss": "mse", "epochs": 10, "batch_size": 32},
                       "lagged_autoencoder": {"optimizer": "adam", "loss": "mse", "epochs": 10, "batch_size": 32},
                       "fusion": {"optimizer": "adam", "loss": "mse", "epochs": 10, "batch_size": 32}}
    normalizer = Normalizer()
    pipeline = Pipeline(train_val_ratio=0.7, normalizer=normalizer, model=model, anomaly_score="residuals")
    pipeline.apply(ts, training_config, early_stopping=None, save_dir="../", show_plot=True)
