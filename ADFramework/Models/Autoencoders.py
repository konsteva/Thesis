import os
import sys
import numpy as np
import json
import zipfile

import torch
import torch.nn as nn
import torch.optim as optim
from keras.regularizers import L1

from tqdm import tqdm

from keras import Sequential, Input
from keras.layers import Dense, Concatenate, BatchNormalization, Activation
from keras.models import model_from_json, load_model
from keras.models import Model as KerasModel
from keras.callbacks import EarlyStopping

from ADFramework.Models.Model import Model
from ADFramework.Utilities import Utils
from ADFramework.Models.Callbacks import CustomVerbosity, DelayedEarlyStopping
from ADFramework.Utilities.TimeSeries import TimeSeries


class VanillaAutoencoder(Model):
    def __init__(self, input_size=None, encoder_layers=None, decoder_layers=None, latent_space=None,
                 encoder_activation="relu", decoder_activation="relu", output_activation="sigmoid"):
        # Sigmoid output activation if values are normalized within [0,1]
        super().__init__("Vanilla Autoencoder")
        if encoder_layers is None:
            encoder_layers = [128, 64]
        if decoder_layers is None:
            decoder_layers = [64, 128]
        if latent_space is None:
            latent_space = 32

        self.is_compiled = False
        self.forecast_window = input_size  # autoencoder input size -> name convention for consistency between different models
        self.encoder_layers = encoder_layers
        self.encoder_activation = encoder_activation
        self.latent_space = latent_space
        self.decoder_layers = decoder_layers
        self.decoder_activation = decoder_activation
        self.output_activation = output_activation
        self.training_config = None
        self.history = None
        self.autoencoder = None

    def fit(self, x_train, x_val, training_config, early_stopping=None):
        """
        :param x_train: 1d ndarray of true timeseries training values
        :param x_val: 1d ndarray of the true timeseries validation allues
        :param training_config: dictionary containing the training hyperparameters:
            "optimizer", "loss", "epochs", "batch_size"
        :return: the training history
        """
        self.training_config = training_config
        if self.forecast_window is None:
            self.forecast_window = Utils.find_period(x_train)

        # set input size and compile - necessary for the first fit
        if not self.is_compiled:
            self.initialize()
            self.autoencoder.compile(optimizer=training_config["optimizer"], loss=training_config["loss"])
            self.is_compiled = True

        # segment input time series with segment size equal to window
        train_segments = Utils.segment_timeseries(x_train, window=self.forecast_window)
        val_segments = Utils.segment_timeseries(x_val, window=self.forecast_window)

        callbacks = [CustomVerbosity()]
        if early_stopping is not None:
            callbacks.append(early_stopping)

        history = self.autoencoder.fit(train_segments, train_segments,
                                       epochs=training_config["epochs"],
                                       batch_size=training_config["batch_size"],
                                       validation_data=(val_segments, val_segments),
                                       verbose=0,
                                       callbacks=callbacks)
        self.history = history.history

        return self.history

    def predict(self, x, reconstruct=False):
        """
        Predict in batches to handle possible memory overflows
        :param x: 1d ndarray of the timeseries values to reconstruct
        :param reconstruct: boolean -> whether to reconstruct predicted segments back to a 1d ndarray
        :return: the reconstructed values
        """
        if not self.is_compiled:
            raise Exception("Make sure the model is fitted before predicting")

        x_segments = Utils.segment_timeseries(x, window=self.forecast_window)
        batch_size = self.training_config["batch_size"]
        pred = np.zeros(shape=(len(x_segments) // batch_size, batch_size, self.forecast_window))
        for i in tqdm(range(0, len(x_segments) // batch_size + 1)):
            start = i * batch_size
            end = min(start + batch_size, len((x_segments)))

            # this is in case the number of segments is a multiple of the batch. In this case start and end are the same
            # and out of bounds creating an exception
            if end - start == 0:
                pred = pred.reshape(-1, self.forecast_window)
                if reconstruct:
                    pred, confints = Utils.average_reconstruct_timeseries_confint(pred)
                    res = np.abs(x[:len(pred)] - pred)
                    return pred, confints, res

                return pred

            batch = x_segments[start:end]
            if i == len(x_segments) // batch_size:
                last_pred = self.autoencoder.predict(batch)
            else:
                pred[i] = self.autoencoder.predict_on_batch(batch)

        pred = np.concatenate([pred.reshape(-1, self.forecast_window), last_pred])
        if reconstruct:
            pred, confints = Utils.average_reconstruct_timeseries_confint(pred)
            res = np.abs(x[:len(pred)] - pred)
            return pred, confints, res

        return pred

    def initialize(self):
        self.autoencoder = Sequential()
        self.autoencoder.add(Input(shape=(self.forecast_window,)))
        for layer in self.encoder_layers:
            self.autoencoder.add(Dense(layer, self.encoder_activation))

        self.autoencoder.add(Dense(self.latent_space, self.decoder_activation))

        for layer in self.decoder_layers:
            self.autoencoder.add(Dense(layer, self.decoder_activation))

        self.autoencoder.add(Dense(self.forecast_window, self.output_activation))

        return self.autoencoder

    def save(self, save_dir, zip_name):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save model architecture to JSON
        model_json = self.autoencoder.to_json()
        with open(os.path.join(save_dir, "model.json"), "w") as json_file:
            json_file.write(model_json)

        # Save model weights
        self.autoencoder.save_weights(os.path.join(save_dir, "model_weights.h5"))

        # Save training history
        with open(os.path.join(save_dir, "history.json"), "w") as json_file:
            json.dump(self.history, json_file)

        # Save additional configurations
        config = {
            "input_size": self.forecast_window,
            "encoder_layers": self.encoder_layers,
            "encoder_activation": self.encoder_activation,
            "latent_space": self.latent_space,
            "decoder_layers": self.decoder_layers,
            "decoder_activation": self.decoder_activation,
            "output_activation": self.output_activation,
            "training_config": self.training_config,
        }
        with open(os.path.join(save_dir, "config.json"), "w") as json_file:
            json.dump(config, json_file)

        # Zip the saved files
        if not zip_name.endswith("zip"):
            zip_name += ".zip"
        zip_path = os.path.join(save_dir, zip_name)
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(os.path.join(save_dir, "model.json"), arcname="model.json")
            zipf.write(os.path.join(save_dir, "model_weights.h5"), arcname="model_weights.h5")
            zipf.write(os.path.join(save_dir, "history.json"), arcname="history.json")
            zipf.write(os.path.join(save_dir, "config.json"), arcname="config.json")

        # Remove the individual files after zipping
        os.remove(os.path.join(save_dir, "model.json"))
        os.remove(os.path.join(save_dir, "model_weights.h5"))
        os.remove(os.path.join(save_dir, "history.json"))
        os.remove(os.path.join(save_dir, "config.json"))

    @classmethod
    def load(cls, model_dir, zip_name):
        zip_path = os.path.join(model_dir, zip_name)

        with zipfile.ZipFile(zip_path, 'r') as zipf:
            zipf.extractall(model_dir)

        # Load configurations
        with open(os.path.join(model_dir, "config.json"), "r") as json_file:
            config = json.load(json_file)

        # Create instance with loaded configurations
        model = cls(input_size=config["input_size"],
                    encoder_layers=config["encoder_layers"],
                    decoder_layers=config["decoder_layers"],
                    latent_space=config["latent_space"],
                    encoder_activation=config["encoder_activation"],
                    decoder_activation=config["decoder_activation"],
                    output_activation=config["output_activation"])

        # Load model architecture
        with open(os.path.join(model_dir, "model.json"), "r") as json_file:
            model_json = json_file.read()
            model.autoencoder = model_from_json(model_json)

        # Load model weights
        model.autoencoder.load_weights(os.path.join(model_dir, "model_weights.h5"))

        # Load training history
        with open(os.path.join(model_dir, "history.json"), "r") as json_file:
            model.history = json.load(json_file)

        # Set the training config if available
        model.training_config = config.get("training_config")
        model.autoencoder.compile(optimizer=model.training_config["optimizer"], loss=model.training_config["loss"])
        model.is_compiled = True

        # Remove the extracted files
        os.remove(os.path.join(model_dir, "model.json"))
        os.remove(os.path.join(model_dir, "model_weights.h5"))
        os.remove(os.path.join(model_dir, "history.json"))
        os.remove(os.path.join(model_dir, "config.json"))

        return model


class _SharedWeightsAutoencoder(nn.Module):
    def __init__(self, input_size, encoder_layers, latent_space):
        super(_SharedWeightsAutoencoder, self).__init__()

        self.encoder = nn.ModuleList()
        in_dim = input_size
        for layer_dim in encoder_layers:
            encoder_layer = nn.Linear(in_dim, layer_dim)
            self.encoder.append(encoder_layer)
            in_dim = layer_dim

        encoder_layer = nn.Linear(in_dim, latent_space)
        self.encoder.append(encoder_layer)

    def forward(self, x):
        # Encoder
        for i in range(len(self.encoder)):
            x = torch.relu(self.encoder[i](x))

        # Decoder
        for i in range(len(self.encoder)-1):
            x = torch.relu(
                torch.nn.functional.linear(x, self.encoder[-i-1].weight.t())
            )

        # Output
        x = torch.sigmoid(
            torch.nn.functional.linear(x, self.encoder[0].weight.t())
        )

        return x


class SharedWeightsAutoencoder(Model):
    def __init__(self, input_size=None, encoder_layers=None, latent_space=None):
        # Sigmoid output activation if values are normalized within [0,1]
        super().__init__("Shared Weights Autoencoder")
        if encoder_layers is None:
            encoder_layers = [128, 64]
        if latent_space is None:

            latent_space = 32

        self.is_compiled = False
        self.forecast_window = input_size  # autoencoder input size -> name convention for consistency between different models
        self.encoder_layers = encoder_layers
        self.latent_space = latent_space
        self.decoder_layers = encoder_layers[::-1]
        self.training_config = None
        self.history = {"train_loss": [], "val_loss": []}
        self.autoencoder = _SharedWeightsAutoencoder(
            self.forecast_window,
            self.encoder_layers,
            self.latent_space
        )

    @staticmethod
    def _batchify(data, batch_size):
        shape = data.shape
        remainder = len(data) - batch_size * (shape[0] // batch_size)
        batches = data[:batch_size * (shape[0] // batch_size)].reshape(-1, batch_size, shape[1])

        if remainder != 0:
            remainder_batch = data[-remainder:].reshape(-1, remainder, shape[1])
        else:
            remainder_batch = np.asarray([])

        return batches, remainder_batch

    def _update_history(self, train_loss, val_loss):
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)

    def fit(self, x_train, x_val, training_config, early_stopping=None):
        """
        :param x_train: 1d ndarray of true timeseries training values
        :param x_val: 1d ndarray of the true timeseries validation allues
        :param training_config: dictionary of training settings with keys "batch size" and "epochs"
        :param early_stopping: PyTorch early stopping
        :return: the training history
        """
        self.training_config = training_config

        # segment input time series with segment size equal to window
        train_segments = Utils.segment_timeseries(x_train, window=self.forecast_window)
        val_segments = Utils.segment_timeseries(x_val, window=self.forecast_window)

        # set training config
        criterion = nn.MSELoss()  # Mean Squared Error for reconstruction loss
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=1e-4)

        # create batch data
        train_batches, train_remainder_batch = self._batchify(train_segments, training_config["batch_size"])
        val_batches, val_remainder_batch = self._batchify(val_segments, training_config["batch_size"])

        # Convert to tensor
        train_batches = torch.from_numpy(train_batches).to(torch.float32)
        val_batches = torch.from_numpy(val_batches).to(torch.float32)
        train_remainder_batch = torch.from_numpy(train_remainder_batch).to(torch.float32)
        val_remainder_batch = torch.from_numpy(val_remainder_batch).to(torch.float32)

        # Training loop with validation
        num_epochs = training_config["epochs"]
        for epoch in range(num_epochs):
            self.autoencoder.train()
            train_loss = 0.0
            for batch_data in train_batches:
                # batch_data = torch.from_numpy(batch_data).to(torch.float32)
                optimizer.zero_grad()

                output = self.autoencoder(batch_data)
                loss = criterion(output, batch_data)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # if there are remaining train segments
            if train_remainder_batch.size != 0:
                # train_remainder_batch = torch.from_numpy(train_remainder_batch).to(torch.float32)
                optimizer.zero_grad()
                output = self.autoencoder(train_remainder_batch)
                loss = criterion(output, train_remainder_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Average training loss over batches
            train_loss /= (len(train_batches) + len(train_remainder_batch))

            # Validation loop
            self.autoencoder.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_data in val_batches:
                    # batch_data = torch.from_numpy(batch_data).to(torch.float32)
                    output = self.autoencoder(batch_data)
                    loss = criterion(output, batch_data)
                    val_loss += loss.item()

                # if there are remaining validation segments
                if val_remainder_batch.size != 0:
                    # val_remainder_batch = torch.from_numpy(val_remainder_batch).to(torch.float32)
                    output = self.autoencoder(val_remainder_batch)
                    loss = criterion(output, val_remainder_batch)
                    val_loss += loss.item()

            # Average validation loss over batches
            val_loss /= (len(val_batches) + len(val_remainder_batch))

            self._update_history(train_loss, val_loss)

            # print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            print(f'\rEpoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}', end='')
            sys.stdout.flush()

            if early_stopping is not None:
                if early_stopping.early_stop(val_loss):
                    break

        return self.history

    def predict(self, x, reconstruct=False):
        """
        Predict in batches to handle possible memory overflows
        :param x: 1d ndarray of the timeseries values to reconstruct
        :param reconstruct: boolean -> whether to reconstruct predicted segments back to a 1d ndarray
        :return: the reconstructed values
        """
        x_segments = Utils.segment_timeseries(x, window=self.forecast_window)
        batch_size = self.training_config["batch_size"]
        x_segments_batch, x_segments_rem_batch = self._batchify(x_segments, batch_size)

        # Convert to tensor
        x_segments_batch = torch.from_numpy(x_segments_batch).to(torch.float32)
        x_segments_rem_batch = torch.from_numpy(x_segments_rem_batch).to(torch.float32)

        pred = np.zeros(shape=(
            len(x_segments) // batch_size,
            batch_size,
            self.forecast_window))

        with torch.no_grad():
            for i, batch in tqdm(enumerate(x_segments_batch)):
                pred[i] = self.autoencoder(batch).numpy()

            if x_segments_rem_batch.size()[0] != 0:
                last_pred = self.autoencoder(x_segments_rem_batch).numpy()
                pred = np.concatenate([pred.reshape(-1, self.forecast_window), last_pred.reshape(-1, self.forecast_window)])
            else:
                pred = pred.reshape(-1, self.forecast_window)

        if reconstruct:
            pred, confints = Utils.average_reconstruct_timeseries_confint(pred)
            res = np.abs(x[:len(pred)] - pred)

            return pred, confints, res

        return pred

    def save(self, save_dir, zip_name):
        pass

    @classmethod
    def load(cls, model_dir, zip_name):
        pass


class _LaggedAutoencoder:
    """ Autoencoder that uses previous period to reconstruct the current. It handles internally the data to take into account the previous period """

    def __init__(self, input_size=None, encoder_layers=None, decoder_layers=None, latent_space=None,
                 encoder_activation="relu", decoder_activation="relu", output_activation="sigmoid", batch_norm=False):
        # Sigmoid output activation if values are normalized within [0,1]
        if encoder_layers is None:
            encoder_layers = [128, 64]
        if decoder_layers is None:
            decoder_layers = [64, 128]
        if latent_space is None:
            latent_space = 32

        self.is_compiled = False
        self.forecast_window = input_size  # autoencoder input size -> name convention for consistency between different models
        self.encoder_layers = encoder_layers
        self.encoder_activation = encoder_activation
        self.latent_space = latent_space
        self.decoder_layers = decoder_layers
        self.decoder_activation = decoder_activation
        self.output_activation = output_activation
        self.batch_norm = batch_norm
        self.training_config = None
        self.history = None
        self.autoencoder = None

    def fit(self, x_train, x_val, training_config, early_stopping=False):
        """
        :param x_train: 1d ndarray of true timeseries training values
        :param x_val: 1d ndarray of the true timeseries validation allues
        :param training_config: dictionary containing the training hyperparameters:
            "optimizer", "loss", "epochs", "batch_size"
        :return: the training history
        """
        self.training_config = training_config
        if self.forecast_window is None:
            self.forecast_window = Utils.find_period(x_train)

        # set input size and compile - necessary for the first fit
        if not self.is_compiled:
            self.initialize()
            self.autoencoder.compile(optimizer=training_config["optimizer"], loss=training_config["loss"])
            self.is_compiled = True

        # segment input time series with segment size equal to window
        """ Here the lagged autoencoder splits the data using two periods  """
        train_segments = Utils.segment_timeseries(x_train, window=2 * self.forecast_window)
        val_segments = Utils.segment_timeseries(x_val, window=2 * self.forecast_window)

        """ Previous period in input and desired output is current period """
        train_previous_period = train_segments[:, :train_segments.shape[1] // 2]
        train_current_period = train_segments[:, train_segments.shape[1] // 2:]

        val_previous_period = val_segments[:, :val_segments.shape[1] // 2]
        val_current_period = val_segments[:, val_segments.shape[1] // 2:]

        callbacks = [CustomVerbosity()]
        if early_stopping:
            early_stopping = EarlyStopping(
                monitor='val_loss',
                min_delta=0.0005,  # Minimum change to qualify as an improvement
                patience=10,
                verbose=1,
                restore_best_weights=True
            )

            early_stopping = DelayedEarlyStopping(early_stopping_callback=early_stopping, start_epoch=30)
            callbacks.append(early_stopping)

        history = self.autoencoder.fit(train_previous_period, train_current_period,
                                       epochs=training_config["epochs"],
                                       batch_size=training_config["batch_size"],
                                       validation_data=(val_previous_period, val_current_period),
                                       verbose=0,
                                       callbacks=callbacks)
        self.history = history.history

        return self.history

    def predict(self, x, reconstruct=False):
        """
        Predict in batches to handle possible memory overflows
        :param x: 1d ndarray of the timeseries values to reconstruct
        :param reconstruct: boolean -> whether to reconstruct predicted segments back to a 1d ndarray
        :return: the reconstructed values
        """
        if not self.is_compiled:
            raise Exception("Make sure the model is fitted before predicting")

        """ Here the lagged autoencoder splits the data using two periods and only the past period """
        x_segments = Utils.segment_timeseries(x, window=2 * self.forecast_window)
        previous_period = x_segments[:, :x_segments.shape[1] // 2]
        current_period = x_segments[:, x_segments.shape[1] // 2:]
        batch_size = self.training_config["batch_size"]
        pred = np.zeros(shape=(len(x_segments) // batch_size, batch_size, self.forecast_window))
        for i in tqdm(range(0, len(x_segments) // batch_size + 1)):
            start = i * batch_size
            end = min(start + batch_size, len(x_segments))

            # this is in case the number of segments is a multiple of the batch. In this case start and end are the same
            # and out of bounds creating an exception
            if end - start == 0:
                pred = pred.reshape(-1, self.forecast_window)
                if reconstruct:
                    pred, confints = Utils.average_reconstruct_timeseries_confint(pred)
                    res = np.abs(x[:len(pred)] - pred)
                return pred, confints, res

            batch = previous_period[start:end]
            if i == len(x_segments) // batch_size:
                last_pred = self.autoencoder.predict(batch)
            else:
                pred[i] = self.autoencoder.predict_on_batch(batch)

        pred = np.concatenate([pred.reshape(-1, self.forecast_window), last_pred])
        if reconstruct:
            pred, confints = Utils.average_reconstruct_timeseries_confint(pred)
            res = np.abs(x[self.forecast_window:] - pred)

        return pred, confints, res

    def initialize(self):
        self.autoencoder = Sequential()
        self.autoencoder.add(Input(shape=(self.forecast_window,)))
        for layer in self.encoder_layers:
            self.autoencoder.add(Dense(layer, kernel_initializer="uniform", use_bias=False))
            if self.batch_norm:
                self.autoencoder.add(BatchNormalization())
            self.autoencoder.add(Activation(self.encoder_activation))

        self.autoencoder.add(Dense(self.latent_space, kernel_initializer="uniform", use_bias=False))
        if self.batch_norm:
            self.autoencoder.add(BatchNormalization())
        self.autoencoder.add(Activation(self.decoder_activation))

        for layer in self.decoder_layers:
            self.autoencoder.add(Dense(layer, kernel_initializer="uniform", use_bias=False))
            if self.batch_norm:
                self.autoencoder.add(BatchNormalization())
            self.autoencoder.add(Activation(self.decoder_activation))

        self.autoencoder.add(Dense(self.forecast_window, kernel_initializer="uniform", use_bias=False))
        if self.batch_norm:
            self.autoencoder.add(BatchNormalization())
        self.autoencoder.add(Activation(self.output_activation))

        return self.autoencoder

    def save(self, save_dir, zip_name):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save model architecture to JSON
        model_json = self.autoencoder.to_json()
        with open(os.path.join(save_dir, "model.json"), "w") as json_file:
            json_file.write(model_json)

        # Save model weights
        self.autoencoder.save_weights(os.path.join(save_dir, "model_weights.h5"))

        # Save training history
        with open(os.path.join(save_dir, "history.json"), "w") as json_file:
            json.dump(self.history, json_file)

        # Save additional configurations
        config = {
            "input_size": self.forecast_window,
            "encoder_layers": self.encoder_layers,
            "encoder_activation": self.encoder_activation,
            "latent_space": self.latent_space,
            "decoder_layers": self.decoder_layers,
            "decoder_activation": self.decoder_activation,
            "output_activation": self.output_activation,
            "training_config": self.training_config,
        }
        with open(os.path.join(save_dir, "config.json"), "w") as json_file:
            json.dump(config, json_file)

        # Zip the saved files
        if not zip_name.endswith("zip"):
            zip_name += ".zip"
        zip_path = os.path.join(save_dir, zip_name)
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(os.path.join(save_dir, "model.json"), arcname="model.json")
            zipf.write(os.path.join(save_dir, "model_weights.h5"), arcname="model_weights.h5")
            zipf.write(os.path.join(save_dir, "history.json"), arcname="history.json")
            zipf.write(os.path.join(save_dir, "config.json"), arcname="config.json")

        # Remove the individual files after zipping
        os.remove(os.path.join(save_dir, "model.json"))
        os.remove(os.path.join(save_dir, "model_weights.h5"))
        os.remove(os.path.join(save_dir, "history.json"))
        os.remove(os.path.join(save_dir, "config.json"))

    @classmethod
    def load(cls, model_dir, zip_name):
        pass


class FusionModel:
    def __init__(self, input_size, layers=None, hidden_activation="relu", output_activation="sigmoid"):
        if layers is None:
            layers = [64]

        self.is_compiled = False
        self.forecast_window = input_size
        self.layers = layers
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.training_config = None
        self.history = None
        self.model = None

    def fit(self, train_inputs, train_outputs, val_inputs, val_outputs, training_config, early_stopping=False):
        """
        :param train_inputs: Training tuple of 1d ndarrays of the first signal to fuse (lagged period prediction) and the second signal to fuse (which is also the desired output)
        :param val_inputs: Validation tuple of 1d ndarrays of the first signal to fuse (lagged period prediction) and the second signal to fuse (which is also the desired output)
        :param training_config: dictionary containing the training hyperparameters: "optimizer", "loss", "epochs", "batch_size"
        :return: the training history
        """
        train_input1, train_input2 = train_inputs
        val_input1, val_input2 = val_inputs

        # Fix sizes (remove one period from beginning) if they have not been handled already
        if len(train_input2) > len(train_input1):
            train_input2 = train_input2[self.forecast_window:]

        if len(train_outputs) > len(train_input1):
            train_outputs = train_outputs[self.forecast_window:]

        if len(val_input2) > len(val_input1):
            val_input2 = val_input2[self.forecast_window:]

        if len(val_outputs) > len(val_input1):
            val_outputs = val_outputs[self.forecast_window:]

        # set input size and compile - necessary for the first fit
        if not self.is_compiled:
            self.initialize()
            self.model.compile(optimizer=training_config["optimizer"], loss=training_config["loss"])
            self.is_compiled = True

        train_input1_segments = Utils.segment_timeseries(train_input1, self.forecast_window)
        train_input2_segments = Utils.segment_timeseries(train_input2, self.forecast_window)
        train_outputs_segments = Utils.segment_timeseries(train_outputs, self.forecast_window)

        val_input1_segments = Utils.segment_timeseries(val_input1, self.forecast_window)
        val_input2_segments = Utils.segment_timeseries(val_input2, self.forecast_window)
        val_outputs_segments = Utils.segment_timeseries(val_outputs, self.forecast_window)

        callbacks = [CustomVerbosity()]
        if early_stopping:
            early_stopping = EarlyStopping(
                monitor='val_loss',
                min_delta=0.0005,  # Minimum change to qualify as an improvement
                patience=10,
                verbose=1,
                restore_best_weights=True
            )

            early_stopping = DelayedEarlyStopping(early_stopping_callback=early_stopping, start_epoch=30)
            callbacks.append(early_stopping)

        history = self.model.fit([train_input1_segments, train_input2_segments], train_outputs_segments,
                                 epochs=training_config["epochs"],
                                 batch_size=training_config["batch_size"],
                                 validation_data=([val_input1_segments, val_input2_segments], val_outputs_segments),
                                 verbose=0,
                                 callbacks=callbacks)
        self.history = history.history

        return self.history

    def predict(self, inputs, output):
        if not self.is_compiled:
            raise Exception("Make sure the model is fitted before predicting")

        input1, input2 = inputs
        if len(input2) > len(input1):
            input2 = input2[self.forecast_window:]

        if len(output) > len(input1):
            output = output[self.forecast_window:]

        input1_segments = Utils.segment_timeseries(input1, self.forecast_window)
        input2_segments = Utils.segment_timeseries(input2, self.forecast_window)

        pred = self.model.predict([input1_segments, input2_segments])
        pred, confints = Utils.average_reconstruct_timeseries_confint(pred)
        res = np.abs(pred - output)

        return pred, confints, res

    def initialize(self):
        inputs = [Input(shape=int(self.forecast_window)), Input(shape=int(self.forecast_window))]
        concatenated = Concatenate(axis=-1)(inputs)

        x = concatenated
        for units in self.layers:
            x = Dense(units, activation=self.hidden_activation)(x)

        output = Dense(self.forecast_window, activation=self.output_activation)(x)
        self.model = KerasModel(inputs=inputs, outputs=output)

        return self.model

    def save(self, model_path):
        """
        Save the trained model and its configuration to files.

        :param model_path: Path to save the model.
        """
        if self.model is None:
            raise ValueError("There is no model to save. Please train the model first.")

        # Save the model architecture and weights
        self.model.save(os.path.join(model_path, "fusion_model.h5"))

        # Save the model configuration
        config = {
            "layers": self.layers,
            "hidden_activation": self.hidden_activation,
            "output_activation": self.output_activation,
            "forecast_window": self.forecast_window,
            "training_config": self.training_config
        }
        with open(os.path.join(model_path, "fusion_model_config.json"), "w") as config_file:
            json.dump(config, config_file)

        print(f"Model and configuration saved to {model_path}")

    def load(self, model_path):
        """
        Load the model and its configuration from files.

        :param model_path: Path to load the model from.
        """
        # Load the model architecture and weights
        self.model = load_model(os.path.join(model_path, "fusion_model.h5"))
        self.is_compiled = True

        # Load the model configuration
        with open(os.path.join(model_path, "fusion_model_config.json"), "r") as config_file:
            config = json.load(config_file)

        self.layers = config["layers"]
        self.hidden_activation = config["hidden_activation"]
        self.output_activation = config["output_activation"]
        self.forecast_window = config["forecast_window"]
        self.training_config = config["training_config"]

        print(f"Model and configuration loaded from {model_path}")


class FusionAutoencoder(Model):
    def __init__(self, input_size, autoencoder_params=None, lagged_autoencoder_params=None,
                 fusion_model_params=None):
        """
        Uses two autoencoders, one to reconstruct current period from previous period and one that reconstructs current
        period from itself. A fusion model is trained to optimize the fusion of the results of the two autoencoders to
        minimize the mse between predicted and true values.

        :param input_size: the size of the input layer (aka the period of the time series)
        :param autoencoder_params: dictionary containing the init parameters for the Vanilla Autoencoder
        :param lagged_autoencoder_params: dictionary containing the init parameters for the Lagged Autoencoder
        :param fusion_model_params: dictionary containing the init parameters for the Vanilla Autoencoder
        """
        super().__init__("Fusion Autoencoder")
        autoencoder_params = autoencoder_params or {}
        lagged_autoencoder_params = lagged_autoencoder_params or {}
        fusion_model_params = fusion_model_params or {}

        self.forecast_window = input_size
        self.autoencoder_params = autoencoder_params
        self.lagged_autoencoder_params = lagged_autoencoder_params
        self.fusion_model_params = fusion_model_params
        self.training_config = None
        self.history = None  # dictionary containing each models history

        self.autoencoder = None
        self.lagged_autoencoder = None
        self.fusion_model = None

        self.is_compiled = False

    def _initialize(self):
        self.autoencoder = VanillaAutoencoder(self.forecast_window, *self.autoencoder_params.values())
        self.lagged_autoencoder = _LaggedAutoencoder(self.forecast_window, *self.lagged_autoencoder_params.values())
        self.fusion_model = FusionModel(self.forecast_window, *self.fusion_model_params.values())

    def fit(self, x_train, x_val, training_config=None, early_stopping=False):
        """
        :param x_train: 1d ndarray of training time series values
        :param x_val: 1d ndarray of validation time series values
        :param training_config: dictionary containing the training configurations (dictionaries as well) for each model.
            The keys of the dictionary must be "autoencoder", "lagged_autoencoder" and "fusion"
        :return:
        """
        if not self.is_compiled:
            self._initialize()
            self.is_compiled = True

        default_training_config = {"optimizer": "adam",
                                   "loss": "mse",
                                   "epochs": 100,
                                   "batch_size": 32}
        if training_config is None:
            training_config["autoencoder"] = default_training_config
            training_config["lagged_autoencoder"] = default_training_config
            training_config["fusion"] = default_training_config

        training_config = {k: (v if v is not None else default_training_config) for k, v in training_config.items()}

        # No segmentation needed here, it is handled appropriately inside each individual model
        autoencoder_history = self.autoencoder.fit(x_train, x_val, training_config["autoencoder"], early_stopping)
        lagged_autoencoder_history = self.lagged_autoencoder.fit(x_train, x_val, training_config["lagged_autoencoder"], early_stopping)

        train_forecasts_lagged, _, _ = self.lagged_autoencoder.predict(x_train, reconstruct=True)
        val_forecasts_lagged, _, _ = self.lagged_autoencoder.predict(x_val, reconstruct=True)

        train_forecasts, _, _ = self.autoencoder.predict(x_train, reconstruct=True)
        val_forecasts, _, _ = self.autoencoder.predict(x_val, reconstruct=True)

        fusion_history = self.fusion_model.fit(
            (train_forecasts_lagged, train_forecasts), x_train,
            (val_forecasts_lagged, val_forecasts), x_val,
            training_config["fusion"],
            early_stopping
        )

        self.history = {"autoencoder": autoencoder_history,
                        "lagged_autoencoder": lagged_autoencoder_history,
                        "fusion": fusion_history}

        return self.history

    def predict(self, x, reconstruct=True):
        forecasts_lagged, _, _ = self.lagged_autoencoder.predict(x, reconstruct=True)
        forecasts, _, _ = self.autoencoder.predict(x, reconstruct=True)

        fused_forecasts, fused_confints, fused_residuals = self.fusion_model.predict([forecasts_lagged, forecasts], x)

        return fused_forecasts, fused_confints, fused_residuals

    def save(self, save_dir, zip_name):
        pass

    @classmethod
    def load(cls, model_dir, zip_name):
        pass


class DenoisingAutoencoder(Model):
    def __init__(self, input_size=None, encoder_layers=None, decoder_layers=None, latent_space=None,
                 encoder_activation="relu", decoder_activation="relu", output_activation="sigmoid"):
        # Sigmoid output activation if values are normalized within [0,1]
        super().__init__("Denoising Autoencoder")
        if encoder_layers is None:
            encoder_layers = [128, 64]
        if decoder_layers is None:
            decoder_layers = [64, 128]
        if latent_space is None:
            latent_space = 32

        self.is_compiled = False
        self.forecast_window = input_size  # autoencoder input size -> name convention for consistency between different models
        self.encoder_layers = encoder_layers
        self.encoder_activation = encoder_activation
        self.latent_space = latent_space
        self.decoder_layers = decoder_layers
        self.decoder_activation = decoder_activation
        self.output_activation = output_activation
        self.training_config = None
        self.history = None
        self.autoencoder = None

        self.noise_percent = None
        self.noise_arr_percent = 0.5

    def fit(self, x_train, x_val, training_config, noise_percent=0.3, early_stopping=None):
        """
        :param x_train: 1d ndarray of true timeseries training values
        :param x_val: 1d ndarray of the true timeseries validation allues
        :param training_config: dictionary containing the training hyperparameters:
            "optimizer", "loss", "epochs", "batch_size"
        :return: the training history
        """
        self.training_config = training_config
        if self.forecast_window is None:
            self.forecast_window = Utils.find_period(x_train)

        # set input size and compile - necessary for the first fit
        if not self.is_compiled:
            self.initialize()
            self.autoencoder.compile(optimizer=training_config["optimizer"], loss=training_config["loss"])
            self.is_compiled = True

        self.noise_percent = noise_percent

        x_train_noise = self.add_gaussian_noise(x_train, self.noise_percent, arr_percentage=self.noise_arr_percent)
        x_val_noise = self.add_gaussian_noise(x_val, self.noise_percent, arr_percentage=self.noise_arr_percent)

        # segment input time series with segment size equal to window
        train_segments = Utils.segment_timeseries(x_train_noise, window=self.forecast_window)
        val_segments = Utils.segment_timeseries(x_val_noise, window=self.forecast_window)

        train_segments_noise = Utils.segment_timeseries(x_train, window=self.forecast_window)
        val_segments_noise = Utils.segment_timeseries(x_val, window=self.forecast_window)

        callbacks = [CustomVerbosity()]
        if early_stopping is not None:
            callbacks.append(early_stopping)

        history = self.autoencoder.fit(train_segments_noise, train_segments,
                                       epochs=training_config["epochs"],
                                       batch_size=training_config["batch_size"],
                                       validation_data=(val_segments_noise, val_segments),
                                       verbose=0,
                                       callbacks=callbacks)
        self.history = history.history

        return self.history

    def predict(self, x, reconstruct=False):
        """
        Predict in batches to handle possible memory overflows
        :param x: 1d ndarray of the timeseries values to reconstruct
        :param reconstruct: boolean -> whether to reconstruct predicted segments back to a 1d ndarray
        :return: the reconstructed values
        """
        if not self.is_compiled:
            raise Exception("Make sure the model is fitted before predicting")

        x_segments = Utils.segment_timeseries(x, window=self.forecast_window)
        batch_size = self.training_config["batch_size"]
        pred = np.zeros(shape=(len(x_segments) // batch_size, batch_size, self.forecast_window))
        for i in tqdm(range(0, len(x_segments) // batch_size + 1)):
            start = i * batch_size
            end = min(start + batch_size, len((x_segments)))

            # this is in case the number of segments is a multiple of the batch. In this case start and end are the same
            # and out of bounds creating an exception
            # TODO: Make this more concise -> almost identical with lines: 113 -  118
            if end - start == 0:
                pred = pred.reshape(-1, self.forecast_window)
                if reconstruct:
                    pred, confints = Utils.average_reconstruct_timeseries_confint(pred)
                    res = np.abs(x[:len(pred)] - pred)
                return pred, confints, res

            batch = x_segments[start:end]
            if i == len(x_segments) // batch_size:
                last_pred = self.autoencoder.predict(batch)
            else:
                pred[i] = self.autoencoder.predict_on_batch(batch)

        pred = np.concatenate([pred.reshape(-1, self.forecast_window), last_pred])
        if reconstruct:
            pred, confints = Utils.average_reconstruct_timeseries_confint(pred)
            res = np.abs(x[:len(pred)] - pred)

        return pred, confints, res

    def initialize(self):
        self.autoencoder = Sequential()
        self.autoencoder.add(Input(shape=(self.forecast_window,)))
        for layer in self.encoder_layers:
            self.autoencoder.add(Dense(layer, self.encoder_activation))

        self.autoencoder.add(Dense(self.latent_space, self.decoder_activation))

        for layer in self.decoder_layers:
            self.autoencoder.add(Dense(layer, self.decoder_activation))

        self.autoencoder.add(Dense(self.forecast_window, self.output_activation))

        return self.autoencoder

    @staticmethod
    def add_gaussian_noise(timeseries, noise_percentage, arr_percentage):
        isTimeseries = False
        if isinstance(timeseries, TimeSeries):
            values = timeseries.values
            isTimeseries = True
        else:
            values = timeseries

        noise = np.random.normal(0, np.std(values), len(timeseries)) * noise_percentage
        not_noisy_idx = np.random.choice(list(range(len(values))), size=int(arr_percentage * len(values)), replace=False)
        noise[not_noisy_idx] = 0

        if isTimeseries:
            noisy_timeseries = TimeSeries(values=timeseries.values + noise, anomalies=timeseries.anomalies, name=timeseries.name, period=timeseries.period)
        else:
            noisy_timeseries = values + noise

        return noisy_timeseries

    def save(self, save_dir, zip_name):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save model architecture to JSON
        model_json = self.autoencoder.to_json()
        with open(os.path.join(save_dir, "model.json"), "w") as json_file:
            json_file.write(model_json)

        # Save model weights
        self.autoencoder.save_weights(os.path.join(save_dir, "model_weights.h5"))

        # Save training history
        with open(os.path.join(save_dir, "history.json"), "w") as json_file:
            json.dump(self.history, json_file)

        # Save additional configurations
        config = {
            "input_size": self.forecast_window,
            "encoder_layers": self.encoder_layers,
            "encoder_activation": self.encoder_activation,
            "latent_space": self.latent_space,
            "decoder_layers": self.decoder_layers,
            "decoder_activation": self.decoder_activation,
            "output_activation": self.output_activation,
            "training_config": self.training_config,
        }
        with open(os.path.join(save_dir, "config.json"), "w") as json_file:
            json.dump(config, json_file)

        # Zip the saved files
        if not zip_name.endswith("zip"):
            zip_name += ".zip"
        zip_path = os.path.join(save_dir, zip_name)
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(os.path.join(save_dir, "model.json"), arcname="model.json")
            zipf.write(os.path.join(save_dir, "model_weights.h5"), arcname="model_weights.h5")
            zipf.write(os.path.join(save_dir, "history.json"), arcname="history.json")
            zipf.write(os.path.join(save_dir, "config.json"), arcname="config.json")

        # Remove the individual files after zipping
        os.remove(os.path.join(save_dir, "model.json"))
        os.remove(os.path.join(save_dir, "model_weights.h5"))
        os.remove(os.path.join(save_dir, "history.json"))
        os.remove(os.path.join(save_dir, "config.json"))

    @classmethod
    def load(cls, model_dir, zip_name):
        zip_path = os.path.join(model_dir, zip_name)

        with zipfile.ZipFile(zip_path, 'r') as zipf:
            zipf.extractall(model_dir)

        # Load configurations
        with open(os.path.join(model_dir, "config.json"), "r") as json_file:
            config = json.load(json_file)

        # Create instance with loaded configurations
        model = cls(input_size=config["input_size"],
                    encoder_layers=config["encoder_layers"],
                    decoder_layers=config["decoder_layers"],
                    latent_space=config["latent_space"],
                    encoder_activation=config["encoder_activation"],
                    decoder_activation=config["decoder_activation"],
                    output_activation=config["output_activation"])

        # Load model architecture
        with open(os.path.join(model_dir, "model.json"), "r") as json_file:
            model_json = json_file.read()
            model.autoencoder = model_from_json(model_json)

        # Load model weights
        model.autoencoder.load_weights(os.path.join(model_dir, "model_weights.h5"))

        # Load training history
        with open(os.path.join(model_dir, "history.json"), "r") as json_file:
            model.history = json.load(json_file)

        # Set the training config if available
        model.training_config = config.get("training_config")
        model.autoencoder.compile(optimizer=model.training_config["optimizer"], loss=model.training_config["loss"])
        model.is_compiled = True

        # Remove the extracted files
        os.remove(os.path.join(model_dir, "model.json"))
        os.remove(os.path.join(model_dir, "model_weights.h5"))
        os.remove(os.path.join(model_dir, "history.json"))
        os.remove(os.path.join(model_dir, "config.json"))

        return model


class SparseAutoencoder(Model):
    def __init__(self, input_size=None, encoder_layers=None, decoder_layers=None, latent_space=None,
                 encoder_activation="relu", decoder_activation="relu", output_activation="sigmoid", l1=None):
        # Sigmoid output activation if values are normalized within [0,1]
        super().__init__("Sparse Autoencoder")
        if encoder_layers is None:
            encoder_layers = [128, 64]
        if decoder_layers is None:
            decoder_layers = [64, 128]
        if latent_space is None:
            latent_space = 32

        self.is_compiled = False
        self.forecast_window = input_size  # autoencoder input size -> name convention for consistency between different models
        self.encoder_layers = encoder_layers
        self.encoder_activation = encoder_activation
        self.latent_space = latent_space
        self.decoder_layers = decoder_layers
        self.decoder_activation = decoder_activation
        self.output_activation = output_activation
        if l1 is None:
            self.l1 = None
        self.l1 = L1(l1)
        self.training_config = None
        self.history = None
        self.autoencoder = None

    def fit(self, x_train, x_val, training_config, early_stopping=None):
        """
        :param x_train: 1d ndarray of true timeseries training values
        :param x_val: 1d ndarray of the true timeseries validation allues
        :param training_config: dictionary containing the training hyperparameters:
            "optimizer", "loss", "epochs", "batch_size"
        :return: the training history
        """
        self.training_config = training_config
        if self.forecast_window is None:
            self.forecast_window = Utils.find_period(x_train)

        # set input size and compile - necessary for the first fit
        if not self.is_compiled:
            self.initialize()
            self.autoencoder.compile(optimizer=training_config["optimizer"], loss=training_config["loss"])
            self.is_compiled = True

        # segment input time series with segment size equal to window
        train_segments = Utils.segment_timeseries(x_train, window=self.forecast_window)
        val_segments = Utils.segment_timeseries(x_val, window=self.forecast_window)

        callbacks = [CustomVerbosity()]
        if early_stopping is not None:
            callbacks.append(early_stopping)

        history = self.autoencoder.fit(train_segments, train_segments,
                                       epochs=training_config["epochs"],
                                       batch_size=training_config["batch_size"],
                                       validation_data=(val_segments, val_segments),
                                       verbose=0,
                                       callbacks=callbacks)
        self.history = history.history

        return self.history

    def predict(self, x, reconstruct=False):
        """
        Predict in batches to handle possible memory overflows
        :param x: 1d ndarray of the timeseries values to reconstruct
        :param reconstruct: boolean -> whether to reconstruct predicted segments back to a 1d ndarray
        :return: the reconstructed values
        """
        if not self.is_compiled:
            raise Exception("Make sure the model is fitted before predicting")

        x_segments = Utils.segment_timeseries(x, window=self.forecast_window)
        batch_size = self.training_config["batch_size"]
        pred = np.zeros(shape=(len(x_segments) // batch_size, batch_size, self.forecast_window))
        for i in tqdm(range(0, len(x_segments) // batch_size + 1)):
            start = i * batch_size
            end = min(start + batch_size, len((x_segments)))

            # this is in case the number of segments is a multiple of the batch. In this case start and end are the same
            # and out of bounds creating an exception
            if end - start == 0:
                pred = pred.reshape(-1, self.forecast_window)
                if reconstruct:
                    pred, confints = Utils.average_reconstruct_timeseries_confint(pred)
                    res = np.abs(x[:len(pred)] - pred)
                return pred, confints, res

            batch = x_segments[start:end]
            if i == len(x_segments) // batch_size:
                last_pred = self.autoencoder.predict(batch)
            else:
                pred[i] = self.autoencoder.predict_on_batch(batch)

        pred = np.concatenate([pred.reshape(-1, self.forecast_window), last_pred])
        if reconstruct:
            pred, confints = Utils.average_reconstruct_timeseries_confint(pred)
            res = np.abs(x[:len(pred)] - pred)

        return pred, confints, res

    def initialize(self):
        self.autoencoder = Sequential()
        self.autoencoder.add(Input(shape=(self.forecast_window,)))
        for i, layer in enumerate(self.encoder_layers):
            # activity_regularizer=L1(0.0001)
            self.autoencoder.add(Dense(layer, self.encoder_activation, activity_regularizer=self.l1))

        # activity_regularizer=L1(0.0001)
        activity_regularizer=None
        self.autoencoder.add(Dense(self.latent_space, self.decoder_activation, activity_regularizer=self.l1))

        for i, layer in enumerate(self.decoder_layers):
            # activity_regularizer=L1(0.0001)
            activity_regularizer=None
            self.autoencoder.add(Dense(layer, self.decoder_activation, activity_regularizer=self.l1))

        self.autoencoder.add(Dense(self.forecast_window, self.output_activation))

        return self.autoencoder

    def save(self, save_dir, zip_name):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save model architecture to JSON
        model_json = self.autoencoder.to_json()
        with open(os.path.join(save_dir, "model.json"), "w") as json_file:
            json_file.write(model_json)

        # Save model weights
        self.autoencoder.save_weights(os.path.join(save_dir, "model_weights.h5"))

        # Save training history
        with open(os.path.join(save_dir, "history.json"), "w") as json_file:
            json.dump(self.history, json_file)

        # Save additional configurations
        config = {
            "input_size": self.forecast_window,
            "encoder_layers": self.encoder_layers,
            "encoder_activation": self.encoder_activation,
            "latent_space": self.latent_space,
            "decoder_layers": self.decoder_layers,
            "decoder_activation": self.decoder_activation,
            "output_activation": self.output_activation,
            "training_config": self.training_config,
        }
        with open(os.path.join(save_dir, "config.json"), "w") as json_file:
            json.dump(config, json_file)

        # Zip the saved files
        if not zip_name.endswith("zip"):
            zip_name += ".zip"
        zip_path = os.path.join(save_dir, zip_name)
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(os.path.join(save_dir, "model.json"), arcname="model.json")
            zipf.write(os.path.join(save_dir, "model_weights.h5"), arcname="model_weights.h5")
            zipf.write(os.path.join(save_dir, "history.json"), arcname="history.json")
            zipf.write(os.path.join(save_dir, "config.json"), arcname="config.json")

        # Remove the individual files after zipping
        os.remove(os.path.join(save_dir, "model.json"))
        os.remove(os.path.join(save_dir, "model_weights.h5"))
        os.remove(os.path.join(save_dir, "history.json"))
        os.remove(os.path.join(save_dir, "config.json"))

    @classmethod
    def load(cls, model_dir, zip_name):
        zip_path = os.path.join(model_dir, zip_name)

        with zipfile.ZipFile(zip_path, 'r') as zipf:
            zipf.extractall(model_dir)

        # Load configurations
        with open(os.path.join(model_dir, "config.json"), "r") as json_file:
            config = json.load(json_file)

        # Create instance with loaded configurations
        model = cls(input_size=config["input_size"],
                    encoder_layers=config["encoder_layers"],
                    decoder_layers=config["decoder_layers"],
                    latent_space=config["latent_space"],
                    encoder_activation=config["encoder_activation"],
                    decoder_activation=config["decoder_activation"],
                    output_activation=config["output_activation"])

        # Load model architecture
        with open(os.path.join(model_dir, "model.json"), "r") as json_file:
            model_json = json_file.read()
            model.autoencoder = model_from_json(model_json)

        # Load model weights
        model.autoencoder.load_weights(os.path.join(model_dir, "model_weights.h5"))

        # Load training history
        with open(os.path.join(model_dir, "history.json"), "r") as json_file:
            model.history = json.load(json_file)

        # Set the training config if available
        model.training_config = config.get("training_config")
        model.is_compiled = True

        # Remove the extracted files
        os.remove(os.path.join(model_dir, "model.json"))
        os.remove(os.path.join(model_dir, "model_weights.h5"))
        os.remove(os.path.join(model_dir, "history.json"))
        os.remove(os.path.join(model_dir, "config.json"))

        return model


def calculate_dynamic_layers(period):
    layer_sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    latent_space = layer_sizes[0]
    input_size = None
    for size in layer_sizes:
        if size <= period // 5:
            latent_space = size
        elif size <= period:
            input_size = size

    decoder_layers = layer_sizes[layer_sizes.index(latent_space) + 1: layer_sizes.index(input_size) + 1]
    encoder_layers = decoder_layers[::-1]

    return encoder_layers, latent_space, decoder_layers