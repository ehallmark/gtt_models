from keras.models import Model
from keras.optimizers import Adam
import keras as k
import numpy as np
np.random.seed(0)
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
np.random.seed(1)


def sentences_to_indices(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()` (described in Figure 4).

    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this.

    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """

    m = X.shape[0]  # number of training examples

    # Initialize X_indices as a numpy matrix of zeros and the correct shape (≈ 1 line)
    X_indices = np.zeros((m, max_len))

    for i in range(m):  # loop over training examples

        # Convert the ith training sentence in lower case and split is into words. You should get a list of words.
        sentence_words = X[i].lower().split()

        # Initialize j to 0
        j = 0

        # Loop over the words of sentence_words
        for w in sentence_words:
            # Set the (i,j)th entry of X_indices to the index of the correct word.
            X_indices[i, j] = word_to_index[w]
            # Increment j to j + 1
            j = j + 1
    return X_indices


def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.

    Arguments:
    word_to_vec_map -- dictionary mapping words to their Word2Vec vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """

    vocab_len = len(word_to_index) + 1  # adding 1 to fit Keras embedding (requirement)
    emb_dim = word_to_vec_map[word_to_vec_map.keys()[0]].shape[0]  # define dimensionality of your word vectors
    print("Found word2vec dimesnsions: ", emb_dim)

    # Initialize the embedding matrix as a numpy array of zeros of shape
    # (vocab_len, dimensions of word vectors = emb_dim)
    emb_matrix = np.zeros((vocab_len, emb_dim))

    # Set each row "index" of the embedding matrix to be the word vector representation of
    # the "index"th word of the vocabulary
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    # Define Keras embedding layer with the correct output/input sizes, make it trainable.
    # Use Embedding(...). Make sure to set trainable=False.
    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)

    # Build the embedding layer, it is required before setting the weights of the embedding layer.
    # Do not modify the "None".
    embedding_layer.build((None,))

    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer


def create_rnn_encoding_model(input_shape, word_to_vec_map, word_to_index,
                              hidden_layer_size=256, compile=True):
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
    sentence_indices = Input(shape=input_shape, dtype=np.int32)

    # Create the embedding layer pretrained with GloVe Vectors (≈1 line)
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)

    # Propagate sentence_indices through your embedding layer, you get back the embeddings
    embeddings = embedding_layer(sentence_indices)

    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a batch of sequences.
    X = LSTM(hidden_layer_size, return_sequences=True)(embeddings)
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X)
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a single hidden state, not a batch of sequences.
    X = LSTM(hidden_layer_size, return_sequences=False)(X)
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X)
    # Propagate X through a Dense layer with softmax activation to get back a batch of 5-dimensional vectors.
    X = Dense(5, activation="softmax")(X)

    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=sentence_indices, outputs=X)
    if compile:
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def load_rnn_encoding_model(model_file, lr=0.001,
                            loss_func='categorical_crossentropy', compile=True):
    print("Using previous model...")
    model = k.models.load_model(model_file, compile=False)
    if compile:
        model.compile(loss=loss_func, optimizer=Adam(lr=lr), metrics=['accuracy'])
    return model


class RnnEncoder:
    def __init__(self, filepath, load_previous_model=True, batch_size=512, word2vec_size=256,
                 embedding_size=64,
                 vocab_size=None, lr=0.001,
                 loss_func='mean_squared_error',
                 callback=None):
        self.filepath = filepath
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.loss_func = loss_func
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
            self.model = create_rnn_encoding_model(embedding_size, word2vec_size, vocab_size, lr=lr, loss_func=loss_func)

    def train(self, inputs, outputs, val_data, epochs=1, shuffle=True, callbacks=None):
        for cnt in range(epochs):
            self.model.fit(inputs, outputs, verbose=1, batch_size=self.batch_size,
                           validation_data=val_data, shuffle=shuffle, callbacks=callbacks)
            if self.callback is not None:
                self.callback()

    def save(self):
        # save
        self.model.save(self.filepath)

    def load(self):
        self.model = load_rnn_encoding_model(self.filepath, lr=self.lr, loss_func=self.loss_func)



