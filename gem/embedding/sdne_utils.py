import numpy as np

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, model_from_json
import tensorflow.keras.regularizers as Reg


def model_batch_predictor(model, X, batch_size):
    n_samples = X.shape[0]
    counter = 0
    pred = None
    while counter < n_samples // batch_size:
        _, curr_pred = \
            model.predict(X[batch_size * counter:batch_size * (counter + 1),
                          :].toarray())
        if counter:
            pred = np.vstack((pred, curr_pred))
        else:
            pred = curr_pred
        counter += 1
    if n_samples % batch_size != 0:
        _, curr_pred = \
            model.predict(X[batch_size * counter:, :].toarray())
        if counter:
            pred = np.vstack((pred, curr_pred))
        else:
            pred = curr_pred
    return pred


def batch_generator_sdne(X, beta, batch_size, shuffle):
    row_indices, col_indices = X.nonzero()
    sample_index = np.arange(row_indices.shape[0])
    number_of_batches = row_indices.shape[0] // batch_size
    counter = 0
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = \
            sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch_v_i = X[row_indices[batch_index], :].toarray()
        X_batch_v_j = X[col_indices[batch_index], :].toarray()
        InData = np.append(X_batch_v_i, X_batch_v_j, axis=1)

        B_i = np.ones(X_batch_v_i.shape)
        B_i[X_batch_v_i != 0] = beta
        B_j = np.ones(X_batch_v_j.shape)
        B_j[X_batch_v_j != 0] = beta
        X_ij = X[row_indices[batch_index], col_indices[batch_index]]
        deg_i = np.sum(X_batch_v_i != 0, 1).reshape((batch_size, 1))
        deg_j = np.sum(X_batch_v_j != 0, 1).reshape((batch_size, 1))
        a1 = np.append(B_i, deg_i, axis=1)
        a2 = np.append(B_j, deg_j, axis=1)
        OutData = [a1, a2, X_ij.T]
        counter += 1
        yield InData, OutData
        if counter == number_of_batches:
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0


def get_encoder(node_num, d, K, n_units, nu1, nu2, activation_fn):
    # Input
    x = Input(shape=(node_num,))
    # Encoder layers
    y = [None] * (K + 1)
    y[0] = x  # y[0] is assigned the input
    for i in range(K - 1):
        y[i + 1] = Dense(n_units[i],
                         activation=activation_fn,
                         kernel_regularizer=Reg.l1_l2(l1=nu1, l2=nu2))(y[i])
    y[K] = Dense(d,
                 activation=activation_fn,
                 kernel_regularizer=Reg.l1_l2(l1=nu1, l2=nu2))(y[K - 1])
    # Encoder model
    encoder = Model(inputs=x, outputs=y[K])
    return encoder


def get_decoder(node_num, d, K,
                n_units, nu1, nu2,
                activation_fn):
    # Input
    y = Input(shape=(d,))
    # Decoder layers
    y_hat = [None] * (K + 1)
    y_hat[K] = y
    for i in range(K - 1, 0, -1):
        y_hat[i] = Dense(n_units[i - 1],
                         activation=activation_fn,
                         kernel_regularizer=Reg.l1_l2(l1=nu1, l2=nu2))(y_hat[i + 1])
    y_hat[0] = Dense(node_num, activation=activation_fn,
                     kernel_regularizer=Reg.l1_l2(l1=nu1, l2=nu2))(y_hat[1])
    # Output
    x_hat = y_hat[0]  # decoder's output is also the actual output
    # Decoder Model
    decoder = Model(inputs=y, outputs=x_hat)
    return decoder


def get_autoencoder(encoder, decoder):
    # Input
    x = Input(shape=(encoder.layers[0].input_shape[0][1],))
    # Generate embedding
    y = encoder(x)
    # Generate reconstruction
    x_hat = decoder(y)
    # Autoencoder Model
    autoencoder = Model(inputs=x, outputs=[x_hat, y])
    return autoencoder


def graphify(reconstruction):
    [n1, n2] = reconstruction.shape
    n = min(n1, n2)
    reconstruction = np.copy(reconstruction[0:n, 0:n])
    reconstruction = (reconstruction + reconstruction.T) / 2
    reconstruction -= np.diag(np.diag(reconstruction))
    return reconstruction


def loadmodel(filename):
    model = None
    try:
        model = model_from_json(open(filename).read())
    except FileNotFoundError or IOError:
        print('Error reading file: {0}. Cannot load previous model'.format(filename))
        exit()
    return model


def loadweights(model, filename):
    try:
        model.load_weights(filename)
    except FileNotFoundError or IOError:
        print('Error reading file: {0}. Cannot load previous weights'.format(filename))
        exit()


def savemodel(model, filename):
    json_string = model.to_json()
    open(filename, 'w').write(json_string)


def saveweights(model, filename):
    model.save_weights(filename, overwrite=True)
