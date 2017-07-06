from dataset import load_dataset
import numpy as np
from models.utils import feature_scaler
from models.neural_bof import NBoF


def evaluate_model(train_data, train_labels, test_data, test_labels, update_bof=False):

    np.random.seed(1)
    # Load data
    train_data, test_data = np.float32(train_data), np.float32(test_data)
    train_labels, test_labels = np.int32(train_labels), np.int32(test_labels)
    train_data, test_data = feature_scaler(train_data, test_data)

    # Train the model
    model = NBoF(n_codewords=8, n_output=2, n_hidden=512, g=0.1, update_bof=update_bof,
                 feature_dimension=train_data.shape[2], eta_V=0.01, eta_W=0.01)
    model.init(train_data)

    n_iters, n_pretrain = 1000, 100
    if update_bof:
        model.fit(train_data, train_labels, batch_size=64, type='finetune', n_iters=n_pretrain, )
        model.fit(train_data, train_labels, batch_size=64, type='full', n_iters=n_iters - n_pretrain, )
    else:
        model.fit(train_data, train_labels, batch_size=64, type='finetune', n_iters=n_iters)

    print "Acc = ", np.sum(test_labels == model.predict(test_data)) / float(test_labels.shape[0])


if __name__ == '__main__':

    # Before running the code, download the EEG dataset (https://archive.ics.uci.edu/ml/datasets/eeg+database)
    # and move it to the data folder

    train_data, train_labels, test_data, test_labels = load_dataset()
    train_data = train_data.transpose((0, 2, 1))
    test_data = test_data.transpose((0, 2, 1))

    print "Regular"
    evaluate_model(train_data, train_labels, test_data, test_labels, update_bof=False)

    print "Neural..."
    evaluate_model(train_data, train_labels, test_data, test_labels, update_bof=True)
