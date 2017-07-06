from os.path import isfile, isdir, join
from os import listdir
import gzip
import numpy as np
import cPickle as pickle
from os.path import exists


def extract_data(train_path='data/SMNI_CMI_TRAIN', test_path='data/SMNI_CMI_TEST'):
    """
    Transforms the EEG dataset into the appropriate format
    :param train_path:
    :param test_path:
    :return:
    """

    def load_from_folder(path):
        files = [join(path, x) for x in listdir(path) if isfile(join(path, x)) and x[-3:] == '.gz']
        eeg_data = []
        for file_path in files:
            cfile = gzip.open(file_path, 'r')
            lines = cfile.readlines()
            cfile.close()
            data = lines[4:]
            data = [x for i, x in enumerate(data) if np.mod(i, 257) != 0]

            values = np.float32([float(x.split()[3]) for x in data]).reshape((64, 256))
            eeg_data.append(values)
        eeg_data = np.float32(eeg_data)
        return eeg_data

    print "Extracting dataset..."
    data = []
    labels = []
    for folder in listdir(train_path):
        if isdir(join(train_path, folder)):
            data.append(load_from_folder(join(train_path, folder)))
            if folder[3] == 'a':
                labels.append(np.ones((data[-1].shape[0],)))
            else:
                labels.append(0 * np.ones((data[-1].shape[0],)))
    train_data = np.float32(np.concatenate(data))
    train_labels = np.float32(np.concatenate(labels))

    data = []
    labels = []
    for folder in listdir(test_path):
        if isdir(join(test_path, folder)):
            data.append(load_from_folder(join(test_path, folder)))
            if folder[3] == 'a':
                labels.append(np.ones((data[-1].shape[0],)))
            else:
                labels.append(0 * np.ones((data[-1].shape[0],)))
    test_data = np.float32(np.concatenate(data))
    test_labels = np.float32(np.concatenate(labels))

    with open('data/dataset.pickle', 'w') as f:
        pickle.dump(train_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(train_labels, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_labels, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_dataset():
    """
    Loads the EEG dataset
    :return:
    """

    if not exists('data/dataset.pickle'):
        extract_data()

    with open('data/dataset.pickle', 'r') as f:
        train_data = pickle.load(f)
        train_labels = pickle.load(f)
        test_data = pickle.load(f)
        test_labels = pickle.load(f)

    return train_data, train_labels, test_data, test_labels
