import pickle
import os
import numpy as np
from tensorflow.keras.utils import Sequence
from scipy.signal.windows import tukey
import matplotlib.pyplot as plt
import librosa

class DataGeneratorPickles(Sequence):

    def __init__(self, data_dir, filename, input_size, cond_size, batch_size=10):
        """
        Initializes a data generator object for the CL1B dataset
          :param filename: the name of the dataset
          :param data_dir: the directory in which data are stored
          :param input_size: the input size
          :param cond_size: the number of conditioning parameter
          :param batch_size: The size of each batch returned by __getitem__
        """
        file_data = open(os.path.normpath('/'.join([data_dir, filename])), 'rb')
        Z = pickle.load(file_data)

        self.cond_size = cond_size
        self.x = np.array(Z['x'][:, :], dtype=np.float32)
        self.y = np.array(Z['y'][:, :], dtype=np.float32)

        # windowing the signal to avoid misalignments
        self.x = self.x * np.array(tukey(self.x.shape[1], alpha=0.000005), dtype=np.float32).reshape(1, -1)
        self.y = self.y * np.array(tukey(self.x.shape[1], alpha=0.000005), dtype=np.float32).reshape(1, -1)

        self.batch_size = batch_size
        
        rep = self.x.shape[1]
        self.x = self.x.reshape(-1)
        self.y = self.y.reshape(-1)
        # remove the last samples if not enough for a batch
        lim = int((self.x.shape[0] / self.batch_size) * self.batch_size)

        self.x = self.x[:lim].reshape(self.batch_size, -1)
        self.y = self.y[:lim].reshape(self.batch_size, -1)

        if self.cond_size == 1:
            self.z = np.array(Z['z'], dtype=np.float32)
            self.z = np.array([0., 0.], dtype=np.float32)
            self.z = self.z.reshape(self.z.shape[0], 1)
            self.z = np.repeat(self.z, rep, axis=-1)
            self.z = self.z.reshape(-1)
            self.z = self.z[:lim].reshape(self.batch_size, -1)

        elif self.cond_size == 3:
            self.z = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
            self.z = np.repeat(self.z, rep, axis=-1).T.reshape(self.batch_size, self.x.shape[1], -1)

        del Z
        self.window = input_size

        # how many iterations are needed
        self.training_steps = (lim // self.batch_size)
        self.total_length = lim
        self.on_epoch_end()

    def on_epoch_end(self):
        # create/reset the vector containing the indices of the batches
        self.indices = np.arange(self.x.shape[1] + self.window)

    def __len__(self):
        # compute the needed number of iterations before conclude one epoch
        return int((self.x.shape[1]))-1-self.window

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
            if i == self.__len__() - 1:
                self.on_epoch_end()

    def __getitem__(self, idx):
        
        # get the indices of the requested batch
        indices = self.indices[idx:(idx + 1)] + self.window
        t = indices[0]
        if self.cond_size != 0:

            X = (np.array(self.x[:, t - self.window: t]))
            Y = (np.array(self.y[:, t-1:t]))
            Z = (np.array(self.z[:, t-1:t]))

            X = X.reshape(-1, self.window, 1)

            return [Z, X[:, :-1, :], X[:, -1:, :]], Y
        else:

            X = (np.array(self.x[:, t - self.window: t]))
            Y = (np.array(self.y[:, t]))
            X = X.reshape(-1, self.window, 1)         
        
            return [X[:, :-1, :], X[:, -1, :].reshape(-1, 1, 1)], Y
