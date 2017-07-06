from sklearn.preprocessing import StandardScaler


def feature_scaler(train_vecs, test_vecs):
    """
    Scales the BoF feature vectors
    :param train_vecs:
    :param test_vecs:
    :return:
    """
    scaler = StandardScaler()
    train_shape = train_vecs.shape
    test_shape = test_vecs.shape
    train_vecs = scaler.fit_transform(train_vecs.reshape((train_shape[0] * train_shape[1], -1)))
    test_vecs = scaler.transform(test_vecs.reshape((test_shape[0] * test_shape[1], -1)))
    train_vecs = train_vecs.reshape((train_shape[0], train_shape[1], -1))
    test_vecs = test_vecs.reshape((test_shape[0], test_shape[1], -1))
    return train_vecs, test_vecs


