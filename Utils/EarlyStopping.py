import numpy as np

from .Checkpoint import SaveCheckpoint


class EarlyStopping(object):
    def __init__(self, min_delta, patience, model_save_path, verbose=1):
        self.min_delta = abs(min_delta)
        self.patience = patience
        self.wait = 0
        self.stopped_epoch = 0
        self.model_save_path = model_save_path
        self.best_epoch = 0
        self.verbose = verbose

    def on_train_begin(self):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = -np.Inf

    def on_epoch_end(self, epoch, current, net):
        if np.greater(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            SaveCheckpoint(net,
                           self.model_save_path,
                           'best',
                           verbose=self.verbose)
            self.best_epoch = epoch
            if self.verbose:
                print('update best {} at epoch {}'.format(self.best, epoch))
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                return True
        return False

    def on_train_end(self):
        if self.stopped_epoch > 0:
            if self.verbose:
                print('Epoch %05d: early stopping' % (self.stopped_epoch))
