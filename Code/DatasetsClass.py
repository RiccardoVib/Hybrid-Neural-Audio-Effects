import pickle
import os
import numpy as np
from tensorflow.keras.utils import Sequence
from Utils import filterAudio
from scipy.signal.windows import tukey
import matplotlib.pyplot as plt


class DataGeneratorPickles(Sequence):

    def __init__(self, data_dir, filename, input_size, cond_size, batch_size=10):
        """
        Initializes a data generator object
          :param data_dir: the directory in which data are stored
          :param output_size: output size
          :param batch_size: The size of each batch returned by __getitem__
        """
        file_data = open(os.path.normpath('/'.join([data_dir, filename])), 'rb')
        Z = pickle.load(file_data)

        self.cond_size = cond_size
        self.x = np.array(Z['x'][:, :], dtype=np.float32)
        self.y = np.array(Z['y'][:, :], dtype=np.float32)
       
        #self.x = np.array(filterAudio(self.x, 30, 22000, 44100), dtype=np.float32)
        #self.y = np.array(filterAudio(self.y, 30, 22000, 44100), dtype=np.float32)
        self.x = self.x * np.array(tukey(self.x.shape[1], alpha=0.005), dtype=np.float32).reshape(1, -1)
        self.y = self.y * np.array(tukey(self.x.shape[1], alpha=0.005), dtype=np.float32).reshape(1, -1)

        self.batch_size = batch_size

        rep = self.x.shape[1]
        self.x = self.x.reshape(-1)
        self.y = self.y.reshape(-1)
        lim = int((self.x.shape[0] / self.batch_size) * self.batch_size)
        self.x = self.x[:lim]
        self.y = self.y[:lim]

        if self.cond_size != 0:
            self.z = np.array(Z['z'], dtype=np.float32)
            #self.z = np.array([0.5, 0.0, 1.0], dtype=np.float32)
            #self.z = np.repeat(self.z, rep, axis=-1)
        del Z
        self.window = input_size
        self.training_steps = (lim // self.batch_size)
        self.total_length = lim
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indices = np.arange(self.x.shape[0] + self.window - 1)

    def __len__(self):
        return int((self.x.shape[0]) / self.batch_size) - 1

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
            if i == self.__len__() - 1:
                self.on_epoch_end()

    def __getitem__(self, idx):
        ## Initializing Batch
        X = []  # np.empty((self.batch_size, 2*self.w))
        Y = []  # np.empty((self.batch_size, self.output_size))
        Z = []  # np.empty((self.batch_size, self.cond_size))
        X2 = []  # np.empty((self.batch_size, self.cond_size))

        # get the indices of the requested batch
        indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size] + self.window
        if self.cond_size != 0:
            for t in range(indices[0], indices[-1] + 1, 1):
                X.append(np.array(self.x[t - self.window: t]).T)
                Y.append(np.array(self.y[t]).T)
                Z.append(np.array(self.z[t]).T)

            X = np.array(X, dtype=np.float32).reshape(-1, self.window, 1)
            Y = np.array(Y, dtype=np.float32)
            Z = np.array(Z, dtype=np.float32)
            return [Z, X[:, :-1, :], X[:, -1, :].reshape(-1, 1, 1)], Y
        else:
            for t in range(indices[0], indices[-1] + 1, 1):
                X.append(np.array(self.x[t - self.window: t]).T)
                Y.append(np.array(self.y[t]).T)
                X2.append(np.array(self.y[t - self.window: t - 1]).T)

            X = np.array(X, dtype=np.float32).reshape(-1, self.window, 1)
            Y = np.array(Y, dtype=np.float32)
            return [X[:, :-1, :], X[:, -1, :].reshape(-1, 1, 1)], Y