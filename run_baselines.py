import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, GRU
from sklearn.preprocessing import StandardScaler
from dataset import load_dataset
from keras.utils.np_utils import to_categorical

def scale_data(train_data, test_data):

    data = train_data.reshape((-1, train_data.shape[2]))
    scaler = StandardScaler()
    scaler.fit(data)

    # Scale the training data
    for i in range(train_data.shape[0]):
        train_data[i] = scaler.transform(train_data[i])

    # Scale the training data
    for i in range(test_data.shape[0]):
        test_data[i] = scaler.transform(test_data[i])

    return train_data, test_data

def evaluate_mlp_model(train_data, train_labels, test_data, test_labels):
    np.random.seed(1)

    # Flatten the input
    train_data = train_data.reshape((-1, 64 * 256))
    test_data = test_data.reshape((-1, 64 * 256))

    # Define the model
    model = Sequential()
    model.add(Dense(512, input_dim=64 * 256))
    model.add(Activation('tanh'))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(train_data, to_categorical(train_labels), batch_size=32, epochs=20, verbose=2)
    train_score, train_acc = model.evaluate(train_data, to_categorical(train_labels), batch_size=32, verbose=0)
    test_score, test_acc = model.evaluate(test_data, to_categorical(test_labels), batch_size=32, verbose=0)
    print "Train acc, test acc = ", train_acc, ", ", test_acc


def evaluate_recurrent_model(train_data, train_labels, test_data, test_labels):
    np.random.seed(1)

    # Define the model
    model = Sequential()
    model.add(GRU(256, input_shape=(256, 64)))
    model.add(Activation('tanh'))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(train_data, to_categorical(train_labels), batch_size=32, epochs=20, verbose=2)
    train_score, train_acc = model.evaluate(train_data, to_categorical(train_labels), batch_size=32, verbose=0)
    test_score, test_acc = model.evaluate(test_data, to_categorical(test_labels), batch_size=32, verbose=0)
    print "Train acc, test acc = ", train_acc, ", ", test_acc


if __name__ == '__main__':

    # Load the dataset
    train_data, train_labels, test_data, test_labels = load_dataset()
    train_data = train_data.transpose((0, 2, 1))
    test_data = test_data.transpose((0, 2, 1))
    train_data, test_data = scale_data(train_data, test_data)

    print "Evaluating MLP ..."
    evaluate_mlp_model(train_data, train_labels, test_data, test_labels)

    print "Evaluating GRU ..."
    evaluate_recurrent_model(train_data, train_labels, test_data, test_labels)
