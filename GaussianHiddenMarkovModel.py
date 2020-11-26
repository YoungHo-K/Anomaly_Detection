import os
import time
import pickle
import datetime
import matplotlib.pyplot as plt

from tqdm import tqdm
from hmmlearn.hmm import GaussianHMM

from ParamsMatrix import *


class GaussianHiddenMarkovModel:

    def __init__(self, window_size, stride):

        self.WINDOW_SIZE = window_size
        self.STRIDE = stride

        self.classifier = None

    def fit(self, X, n_states=2, n_iter=100, tol=1e-2, algorithm='viterbi', covariance_type='spherical', model_path=None):

        X = self._generate_window(X)

        print('[INFO] Train Gaussian Hidden Markov Model')
        print('     - Algorithm             : {0}'.format(algorithm))
        print('     - Number of States      : {0}'.format(n_states))
        print('     - Covariance Type       : {0}'.format(covariance_type))
        print('     - Tolerance             : {0}'.format(tol))
        print('     - Number of iteration   : {0}'.format(n_iter))

        self.classifier = GaussianHMM(n_components=n_states, covariance_type=covariance_type, algorithm=algorithm,
                                      startprob_prior=start_prob, transmat_prior=trans_prob, means_prior=mean_,
                                      covars_prior=covar_, n_iter=n_iter, random_state=4570, verbose=True)

        lengths = X.shape[0] // self.WINDOW_SIZE
        lengths = [self.WINDOW_SIZE] * lengths

        start_time = time.time()
        self.classifier.fit(X, lengths)
        elapsed = time.time() - start_time
        print('[INFO] Learning Times : {0}'.format(elapsed))
        print('----------------------------------------------------------')

        if model_path is not None:
            if not os.path.exists(model_path):
                os.makedirs(model_path)

            date_time = datetime.datetime.now()
            month, day, hour, minute = date_time.month, date_time.day, date_time.hour, date_time.minute
            dst_model_name = 'GHMM_W{0}_S{1}_K{2}_C[{3}]_T[{4}{5}{6}{7}].pkl'.format(
                self.WINDOW_SIZE, self.STRIDE, n_states, covariance_type, month, day, hour, minute
            )

            dst_model_path = os.path.join(model_path, dst_model_name)
            with open(dst_model_path, 'wb') as file_descriptor:
                pickle.dump(self.classifier, file_descriptor)

    def _generate_window(self, X):

        print('[INFO] Generate Time Window Data')
        print('     - Window Size   : {0}'.format(self.WINDOW_SIZE))
        print('     - Stride        : {0}'.format(self.STRIDE))
        print('     - Data Shape    : {0}'.format(X.shape))

        window_list = list()
        for index in tqdm(range(0, X.shape[0] - self.WINDOW_SIZE + 1, self.STRIDE)):
            window = X[index: index + self.WINDOW_SIZE]

            if window.shape[0] != self.WINDOW_SIZE:
                break

            window_list.extend(window)

        window_arr = np.array(window_list)

        return window_arr

    def predict(self, X, y, threshold=0.1, model_path=None):

        if model_path is not None:
            with open(model_path, 'rb') as file_descriptor:
                self.classifier = pickle.load(file_descriptor)

        assert self.classifier is not None, '[ERROR] You must set the classifier first'

        y_probability_list = list()
        for index in tqdm(range(0, X.shape[0] - self.WINDOW_SIZE + 1, 1)):
            window = X[index: index + self.WINDOW_SIZE]
            y_probability_list.append(self.classifier.score(window))

        y_probability_arr = np.array(y_probability_list)

        window_last_index = self.WINDOW_SIZE - 1
        y_pred = np.zeros(y.shape[0])
        for index, probability in enumerate(y_probability_arr):
            if probability < threshold:
                y_pred[window_last_index + index] = 1

        y_pred = np.array(y_pred)

        return y_pred

    def plot_prediction(self, X, y=None, model_path=None, figure_save_path=None):

        if model_path is not None:
            with open(model_path, 'rb') as file_descriptor:
                self.classifier = pickle.load(file_descriptor)

        assert self.classifier is not None, '[ERROR] You must set the classifier first'

        window_last_index = self.WINDOW_SIZE - 1
        y_probability_arr = np.ones(X.shape[0])
        for index in tqdm(range(0, X.shape[0] - self.WINDOW_SIZE + 1, 1)):
            window = X[index: index + self.WINDOW_SIZE]
            y_probability_arr[window_last_index + index] = self.classifier.score(window)

        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.grid(b=True, axis='y', color='gray', alpha=0.3, linestyle='--')

        X_arr = np.array(range(0, y_probability_arr.shape[0]))

        ax.title.set_text('Log Probability')
        ax.plot(X_arr, y_probability_arr, linewidth=0.8, color='dimgray')
        if y is not None:
            anomaly_index = y == 1
            ax.scatter(X_arr[np.where(anomaly_index)], y_probability_arr[np.where(anomaly_index)], s=5, c='red', label='Anomaly')
            plt.legend(loc='upper right')

        plt.tight_layout()
        if figure_save_path is None:
            plt.show()
        else:
            if not os.path.exists(figure_save_path):
                os.makedirs(figure_save_path)

            date_time = datetime.datetime.now()
            month, day, hour, minute = date_time.month, date_time.day, date_time.hour, date_time.minute
            dst_model_name = 'GHMM_plot_probability_T[{0}{1}{2}{3}].png'.format(month, day, hour, minute)

            dst_figure_path = os.path.join(figure_save_path, dst_model_name)
            plt.savefig(dst_figure_path)


if __name__ == '__main__':

    X = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
    y = np.zeros(4)

    model = GaussianHiddenMarkovModel(window_size=4, stride=1)

    model.fit(X, n_states=2, n_iter=100, tol=1e-3, algorithm='viterbi', covariance_type='spherical')

    model.plot_prediction(X, y)




