from collections import Counter

import numpy as np


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_idx = np.argsort(distances)[: self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_neighbor_labels = [self.y_train[i] for i in k_idx]
        # return the most common class label
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]


if __name__ == "__main__":
    # Imports
    from matplotlib.colors import ListedColormap
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    import librosa
    import os
    import numpy as np

    cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    current_directory = os.getcwd()
    MFCC_NUM = 10
    MFCC_MAX_LEN = 100

    data_mfccs = []
    data_labels = []
    base_array = np.zeros((MFCC_NUM, MFCC_MAX_LEN))

    # label  = ['A','B','C']
    label  = ['A','B','C','D','E']
    for idx, huruf in enumerate(label):
        audio_path = current_directory + "/suara/" + huruf
        
        onlyfilesdata = [f for f in os.listdir(audio_path) if os.path.isfile(os.path.join(audio_path, f))]
        for suara in onlyfilesdata:
            x, sr = librosa.load(audio_path + '/' + suara)
            mfccs = librosa.feature.mfcc(y=x, n_mfcc=MFCC_NUM)

            len_audio = min(len(mfccs[0]), MFCC_MAX_LEN)
            new_mfcc = base_array
            new_mfcc[:, :len_audio] = mfccs[:, :len_audio]

            data_mfccs.append(new_mfcc)
            data_labels.append(idx)

    data_mfccs = np.array(data_mfccs)
    data_mfccs = np.reshape(data_mfccs, (len(data_mfccs), MFCC_NUM*MFCC_MAX_LEN))
    data_labels = np.array(data_labels)

    X, y = data_mfccs, data_labels

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    # y = scaler.transform(y)

    # iris = datasets.load_iris()
    # X, y = iris.data, iris.target
    # print(X.shape)
    # print(y.shape)
    # exit()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    k = 3
    clf = KNN(k=k)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print("KNN classification accuracy", accuracy(y_test, predictions))