from keras.callbacks import Callback, EarlyStopping


class CustomVerbosity(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        print(f"\rEpoch {epoch + 1}/{self.params['epochs']} - ", end="")
        for key, value in logs.items():
            print(f"{key}: {value:.4f} - ", end="")

        # ignore if it is the last epoch
        if epoch != self.params['epochs'] - 1:
            print("\b\b\b\b", end="", flush=True)  # remove extra characters from the end of the line


class DelayedEarlyStopping(Callback):
    def __init__(self,
                 monitor='val_loss',
                 min_delta=0.001,
                 patience=10,
                 verbose=1,
                 restore_best_weights=True,
                 start_epoch=20):
        """
        :param monitor: Loss to monitor train or val
        :param min_delta: Minimum change to qualify as an improvement
        :param patience: Number of iterations to stop after not improving for a
        :param verbose: Verbosity of callback
        :param restore_best_weights: keep model parameters with the least monitor loss
        :param start_epoch: Iteration to start the early stopping (the additional parameter that does not exist in regular EarlyStopping class)
        """
        super(DelayedEarlyStopping, self).__init__()
        self.early_stopping_callback = EarlyStopping(
            monitor=monitor,
            min_delta=min_delta,  # Minimum change to qualify as an improvement
            patience=patience,
            verbose=verbose,
            restore_best_weights=restore_best_weights
        )
        self.start_epoch = start_epoch

    def on_train_begin(self, logs=None):
        # Pass the training start to the EarlyStopping callback
        self.early_stopping_callback.model = self.model
        self.early_stopping_callback.on_train_begin(logs)

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.start_epoch:
            # Pass the epoch end call to the EarlyStopping callback after start_epoch
            self.early_stopping_callback.on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        # Pass the training end to the EarlyStopping callback
        self.early_stopping_callback.on_train_end(logs)


class TorchEarlyStopping:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False