import numpy as np
import lasagne
import theano
import theano.tensor as T
import theano.gradient
import sklearn.cluster as cluster
import theano.gradient

floatX = theano.config.floatX

class Learner:
    def __init__(self):
        pass

    def fit(self, train_data, train_labels, batch_size=50, type='finetune', n_iters=10000):

        idx = np.arange(train_labels.shape[0])
        n_batches = int(len(idx) / batch_size)
        iter_counter = 0
        while True:
            np.random.shuffle(idx)
            loss = 0

            if iter_counter > n_iters:
                break

            for i in range(n_batches):
                iter_counter += 1
                cur_idx = idx[i * batch_size:(i + 1) * batch_size]
                cur_data = train_data[cur_idx]
                cur_labels = train_labels[cur_idx]

                if type == 'finetune':
                    cur_loss = self.finetune_fn(cur_data, cur_labels)
                else:
                    cur_loss = self.train_fn(cur_data, cur_labels)

                loss += cur_loss * batch_size

                if iter_counter > n_iters:
                    break

            if iter_counter > n_iters:
                break

            if n_batches * batch_size < len(idx):
                iter_counter += 1
                cur_idx = idx[n_batches * batch_size:]
                cur_data = train_data[cur_idx]
                cur_labels = train_labels[cur_idx]

                if type == 'finetune':
                    loss += self.finetune_fn(cur_data, cur_labels) * len(cur_idx)
                else:
                    loss += self.train_fn(cur_data, cur_labels) * len(cur_idx)

            print "Epoch loss = ", loss / len(idx)

    def predict(self, data, batch_size=128):
        labels = np.zeros((len(data),))
        n_batches = int(len(data) / batch_size)

        for i in range(n_batches):
            cur_data = data[i * batch_size:(i + 1) * batch_size]
            labels[i * batch_size:(i + 1) * batch_size] = self.predict_fn(cur_data)
        if n_batches * batch_size < len(data):
            cur_data = data[n_batches * batch_size:]
            labels[n_batches * batch_size:] = self.predict_fn(cur_data)
        return labels


class NBoF(Learner):
    """
    Implements the Neural BoF model
    """

    def init(self, data):
        self.bow.initialize_dictionary(data)

    def __init__(self, n_codewords=256, eta=0.001, eta_V=0.001, eta_W=0.01, g=0.1, update_bof=True,
                 n_hidden=512, n_output=3, feature_dimension=144, ):
        Learner.__init__(self)

        self.n_output = n_output

        # Input variables
        input = T.ftensor3('input_data')
        labels = T.ivector('labels')

        # Neural BoF input Layer
        self.bow = NBoFInputLayer(g=g, feature_dimension=feature_dimension, n_codewords=n_codewords)
        network_input = self.bow.sym_histograms(input)
        self.n_len = n_codewords

        # Define the MLP
        network = lasagne.layers.InputLayer(shape=(None, self.n_len), input_var=network_input)
        network = lasagne.layers.DenseLayer(network, n_hidden, nonlinearity=lasagne.nonlinearities.elu,
                                            W=lasagne.init.Orthogonal())
        network = lasagne.layers.DenseLayer(network, n_output, nonlinearity=lasagne.nonlinearities.softmax,
                                            W=lasagne.init.Orthogonal())

        params = lasagne.layers.get_all_params(network, trainable=True)
        prediction = lasagne.layers.get_output(network)
        prediction = T.clip(prediction, 0.00001, 0.99999)
        loss = T.sum(lasagne.objectives.categorical_crossentropy(prediction, labels))

        updates = lasagne.updates.adam(loss, params, learning_rate=eta)
        self.finetune_fn = theano.function(inputs=[input, labels], outputs=loss, updates=updates)

        if update_bof:
            dictionary_grad = T.grad(loss, self.bow.V)
            dictionary_grad = T.switch(T.isnan(dictionary_grad), 0, dictionary_grad)
            updates_V = lasagne.updates.adam(loss_or_grads=[dictionary_grad], params=[self.bow.V], learning_rate=eta_V)
            updates.update(updates_V)

            W_grad = T.grad(loss, self.bow.W)
            W_grad = T.switch(T.isnan(W_grad), 0, W_grad)
            updates_sigma = lasagne.updates.adam(loss_or_grads=[W_grad], params=[self.bow.W], learning_rate=eta_W)
            updates.update(updates_sigma)

        self.train_fn = theano.function(inputs=[input, labels], outputs=loss, updates=updates)
        self.predict_fn = theano.function(inputs=[input], outputs=T.argmax(prediction, axis=1))


class NBoFInputLayer:
    """
    Defines a Neural BoF input layer
    """

    def __init__(self, g=0.1, feature_dimension=89, n_codewords=16):
        """
        Intializes the Neural BoF object
        :param g: defines the softness of the quantization
        :param feature_dimension: dimension of the feature vectors
        :param n_codewords: number of codewords / RBF neurons to be used
        """

        self.Nk = n_codewords
        self.D = feature_dimension

        # RBF-centers / codewords
        V = np.random.rand(self.Nk, self.D).astype(dtype=floatX)
        self.V = theano.shared(value=V, name='V', borrow=True)
        # Input weights for the RBF neurons
        self.W = theano.shared(value=np.ones((self.Nk, self.D), dtype=floatX) / g, name='W')

        # Tensor of input objects (n_objects, n_features, self.D)
        self.X = T.tensor3(name='X', dtype=floatX)

        # Feature matrix of an object (n_features, self.D)
        self.x = T.matrix(name='x', dtype=floatX)

    def sym_histogram(self, X):
        """
        Computes a soft-quantized histogram of a set of feature vectors (X is a matrix).
        :param X: matrix of feature vectors
        :return:
        """
        distances = sym_distance_matrix(X, self.V, self.W)
        membership = T.nnet.softmax(-distances)
        histogram = T.mean(membership, axis=0)
        return histogram

    def sym_histograms(self, X):
        """
        Encodes a set of objects (X is a tensor3)
        :param X: tensor3 containing the feature vectors for each object
        :return:
        """
        histograms, updates = theano.map(self.sym_histogram, X)
        return histograms

    def initialize_dictionary(self, X, max_iter=100, redo=5, n_samples=50000, n_objects=100000):
        """
        Uses the vectors in X to initialize the dictionary
        """
        # Sample objects
        idx = np.random.permutation(X.shape[0])[:n_objects]
        X = X[idx].reshape(-1, X.shape[2])
        features = X[np.random.permutation(X.shape[0])[:n_samples]]

        print "Clustering feature vectors..."
        features = np.float64(features)
        V = cluster.k_means(features, n_clusters=self.Nk, max_iter=max_iter, n_init=redo, n_jobs=1)
        self.V.set_value(np.asarray(V[0], dtype=theano.config.floatX))


def sym_distance_matrix(A, V, W):
    """
    Calculates the distances between the feature vectors in A and the codewords in V (weighted by W)
    :param A: the matrix that contains the feature vectors
    :param V: the matrix that contains the codewords / RBF neurons centers
    :param W: weight matrix (if W is set to 1, then the regular distance matrix is computed)
    :return:
    """

    def row_dist(t, w):
        D = (w * (A - t)) ** 2
        D = T.sum(D, axis=1)
        D = T.maximum(D, 0)
        D = T.sqrt(D)
        return D

    D, _ = theano.map(fn=row_dist, sequences=[V, W])
    return D.T
