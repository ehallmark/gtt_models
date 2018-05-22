from keras.optimizers import Adam
import keras as k
import numpy as np
from keras.models import Model
from keras.layers import Layer, Dense, Input, Flatten, Dropout, LSTM, Concatenate, Activation, Dot, Reshape, Embedding, Masking
import pandas as pd
from keras.callbacks import LearningRateScheduler
from src.models.Word2VecModel import Word2Vec

np.random.seed(1)


def pretrained_embedding_layer(emb_matrix, input_length):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.

    Arguments:
    word_to_vec_map -- dictionary mapping words to their Word2Vec vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """
    vocab_len = emb_matrix.shape[0]
    # vocab_len = vocab_len + 1  # adding 1 to fit Keras embedding (requirement)
    emb_dim = emb_matrix.shape[1]  # define dimensionality of your word vectors
    print("Found word2vec dimensions: ", emb_dim)

    # Use Embedding(...). Make sure to set trainable=False.
    embedding_layer = Embedding(vocab_len, emb_dim, input_length=input_length, mask_zero=False, trainable=False)

    # Build the embedding layer, it is required before setting the weights of the embedding layer.
    # Do not modify the "None".
    embedding_layer.build((None,))

    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer


def create_rnn_encoding_model(Fcpc, Fx, Tx, cpc2vec_data, word2vec_data, embedding_size, hidden_layer_size=128,
                              optimizer=Adam(0.001), loss_func='categorial_crossentropy',
                              compile=True):
    """
    Function creating the Emojify-v2 model's graph.

    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    model -- a model instance in Keras
    """
    print("Creating new rnn encoding model...")
    # Define sentence_indices as the input of the graph, it should be of shape input_shape
    #  and dtype 'int32' (as it contains indices).
    x1_orig = Input(shape=(Tx, 1), dtype=np.int32)
    x2_orig = Input(shape=(Tx, 1), dtype=np.int32)
    cpc_orig = Input(shape=(1,), dtype=np.int32)

    # Propagate sentence_indices through your embedding layer, you get back the embeddings
    embedding_layer = pretrained_embedding_layer(word2vec_data, Tx)
    cpc_embedding_layer = pretrained_embedding_layer(cpc2vec_data, 1)
    x1 = embedding_layer(x1_orig)
    x2 = embedding_layer(x2_orig)
    cpc = cpc_embedding_layer(cpc_orig)
    print("Embedding shape before: ", x1.shape)
    x1 = Reshape((Tx, Fx))(x1)
    x2 = Reshape((Tx, Fx))(x2)
    cpc = Flatten()(cpc)
    print("Embedding shape after: ", x1.shape)

    lstm_w2v = LSTM(hidden_layer_size, activation='tanh', return_sequences=False)
    dense_cpc = Dense(hidden_layer_size, activation='tanh')
    embedding_dense = Dense(embedding_size, activation='tanh')

    x1 = lstm_w2v(x1)
    #  x1 = Dropout(0.2)(x1)  # dropout layer if desired
    x1 = embedding_dense(x1)
    x2 = lstm_w2v(x2)
    #  x2 = Dropout(0.2)(x2)
    x2 = embedding_dense(x2)
    cpc = dense_cpc(cpc)
    #  cpc = Dropout(0.2)(cpc)
    cpc = embedding_dense(cpc)

    dot = Dot(-1, False)([x1, x2])
    dot2 = Dot(-1, False)([x1, cpc])
    y = Dense(1, activation='sigmoid')

    y1 = y(dot)
    y2 = y(dot2)


    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=[x1_orig, x2_orig, cpc_orig], outputs=[y1, y2])
    if compile:
        model.compile(loss=loss_func, optimizer=optimizer, metrics=['accuracy'])
    return model


def load_rnn_encoding_model(model_file, lr=0.001,
                            loss_func='categorical_crossentropy', compile=True):
    print("Using previous model...")
    model = k.models.load_model(model_file, compile=False)
    if compile:
        model.compile(loss=loss_func, optimizer=Adam(lr=lr), metrics=['accuracy'])
    return model


class RnnEncoder:
    def __init__(self, filepath, load_previous_model=True, cpc2vec_size=64, word2vec_data=None, cpc2vec_data=None, batch_size=512, word2vec_size=256,
                 max_len=128,
                 embedding_size=64,
                 decay = None,
                 lr=0.001,
                 loss_func='mean_squared_error',
                 callback=None):
        self.filepath = filepath
        self.embedding_size = embedding_size
        self.cpc2vec_data = cpc2vec_data
        self.batch_size = batch_size
        self.loss_func = loss_func
        self.decay = decay
        self.word2vec_size = word2vec_size
        self.lr = lr
        self.callback = callback
        self.model = None
        if load_previous_model:
            try:
                self.load()
            except:
                print('Could not fine previous model... Creating new one now.')
        if self.model is None:
            self.model = create_rnn_encoding_model(
                cpc2vec_size,
                word2vec_size, max_len,
                optimizer=Adam(lr=lr, decay=decay),
                embedding_size=embedding_size,
                word2vec_data=word2vec_data,
                cpc2vec_data=cpc2vec_data,
                loss_func=loss_func
            )

    def train(self, x1, x2, cpc, y, validation_data, epochs=1, shuffle=True, callbacks=None):
        # train
        self.model.fit([x1, x2, cpc], [y, y], epochs=epochs, validation_data=validation_data,
                       batch_size=self.batch_size, shuffle=shuffle, callbacks=callbacks)
        if self.callback is not None:
            self.callback()


    def save(self):
        # save
        self.model.save(self.filepath)

    def load(self):
        self.model = load_rnn_encoding_model(self.filepath, lr=self.lr, loss_func=self.loss_func)


vocab_vector_file_txt = '/home/ehallmark/Downloads/word2vec256_vectors.txt'
vocab_vector_file_h5 = '/home/ehallmark/Downloads/word2vec256_vectors.h5'  # h5 extension faster?
vocab_index_file = '/home/ehallmark/Downloads/word2vec256_index.txt'
model_file_32 = '/home/ehallmark/data/python/w2v_cpc_rnn_model_keras32.h5'
model_file_64 = '/home/ehallmark/data/python/w2v_cpc_rnn_model_keras64.h5'
model_file_128 = '/home/ehallmark/data/python/w2v_cpc_rnn_model_keras128.h5'
vocab_size = 477909


def get_data():
    x1 = pd.read_csv('/home/ehallmark/Downloads/word_cpc_rnn_keras_x1.csv', sep=',')
    x2 = pd.read_csv('/home/ehallmark/Downloads/word_cpc_rnn_keras_x2.csv', sep=',')
    cpc = pd.read_csv('/home/ehallmark/Downloads/word_cpc_rnn_keras_cpc.csv', sep=',')
    y = pd.read_csv('/home/ehallmark/Downloads/word_cpc_rnn_keras_y.csv', sep=',')

    num_test = 25000
    x1 = np.array(x1)
    x2 = np.array(x2)
    cpc = np.array(cpc)
    y = np.array(y)

    cpc = cpc.reshape((cpc.shape[0], cpc.shape[1]))
    x1 = x1.reshape((x1.shape[0], x1.shape[1], 1))
    x2 = x2.reshape((x2.shape[0], x2.shape[1], 1))

    cpc_val = cpc[:num_test]
    x1_val = x1[:num_test]
    x2_val = x2[:num_test]
    y_val = y[:num_test]
    cpc = cpc[num_test:]
    x1 = x1[num_test:]
    x2 = x2[num_test:]
    y = y[num_test:]

    return ((x1, x2, cpc), y), ([x1_val, x2_val, cpc_val], [y_val, y_val.copy()])


def sample_data(x1, x2, cpc, y, n):
    indices = np.random.choice(cpc.shape[0], n)
    return np.take(x1, indices, 0), np.take(x2, indices, 0), np.take(cpc, indices), np.take(y, indices)


if __name__ == "__main__":
    load_previous_model = False
    learning_rate = 0.0001
    min_learning_rate = 0.000001
    decay = 0
    batch_size = 128
    epochs = 10
    samples_per_epoch = 1000000
    word2vec_size = 256

    embedding_size_to_file_map = {
        #32: model_file_32,
        #64: model_file_64
        128: model_file_128
    }
    scheduler = LearningRateScheduler(lambda n: max(min_learning_rate, learning_rate/(max(1, n*5))))

    cpc2vec_dim = 64
    print('Loading cpc2vec...')
    w2v_model_file_64 = '/home/ehallmark/data/python/cpc_sim_model_keras_word2vec_' + str(cpc2vec_dim) + '.h5'
    cpc2vec = Word2Vec(w2v_model_file_64, load_previous_model=True,
                       loss_func='mean_squared_error', lr=learning_rate)

    print("Num layers: ", len(cpc2vec.model.layers))
    cpc2vec_weights = None
    for layer in cpc2vec.model.layers:
        if isinstance(layer, Embedding):
            print("Found Embedding Layer: ", layer)
            cpc2vec_weights = layer.get_weights()[0]
            print("Weights Shape: ", cpc2vec_weights.shape)

    print('Loading word2vec model...')
    try:
        word2vec_data = np.load(vocab_vector_file_h5)
    except FileNotFoundError:
        print('defaulting to .txt extension')
        word2vec_data = np.loadtxt(vocab_vector_file_txt)

    print('Getting data...')
    (data, data_val) = get_data()

    data, y = data
    x1, x2, cpc = data

    print('Training model...')
    for vector_dim, model_file in embedding_size_to_file_map.items():
        encoder = RnnEncoder(model_file, load_previous_model=load_previous_model,
                             word2vec_size=word2vec_size,
                             batch_size=batch_size, loss_func='mean_squared_error',
                             embedding_size=vector_dim, lr=learning_rate, decay=decay,
                             max_len=x1.shape[1],
                             cpc2vec_data=cpc2vec_weights,
                             word2vec_data=word2vec_data)
        print("Model Summary: ", encoder.model.summary())
        print("Starting to train model with embedding_size: ", vector_dim)
        for i in range(epochs):
            _x1, _x2, _cpc, _y = sample_data(x1, x2, cpc, y, samples_per_epoch)
            model = encoder.model
            model.fit([_x1, _x2, _cpc], [_y, _y.copy()], epochs=epochs, validation_data=data_val,
                      batch_size=batch_size, shuffle=False, callbacks=[scheduler])
            encoder.save()
