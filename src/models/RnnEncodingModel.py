from keras.optimizers import Adam
import keras as k
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation, Dot, Reshape, Embedding, Masking
from src.attention.AttentionModel import AttentionModelCreator
np.random.seed(1)


def pretrained_embedding_layer(emb_matrix):
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
    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)

    # Build the embedding layer, it is required before setting the weights of the embedding layer.
    # Do not modify the "None".
    embedding_layer.build((None,))

    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer


def create_rnn_encoding_model(Fx, Tx, word2vec_data, embedding_size,
                              hidden_layer_size=256, optimizer=Adam(0.001), loss_func='categorial_crossentropy',
                              e1=32,e2=16,
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
    Ty = 1
    Fy = embedding_size
    # Define sentence_indices as the input of the graph, it should be of shape input_shape
    #  and dtype 'int32' (as it contains indices).
    x1_orig = Input(shape=(Tx, 1), dtype=np.int32)
    x2_orig = Input(shape=(Tx, 1), dtype=np.int32)

    # Propagate sentence_indices through your embedding layer, you get back the embeddings
    embedding_layer = pretrained_embedding_layer(word2vec_data)
    x1 = embedding_layer(x1_orig)
    x2 = embedding_layer(x2_orig)
    print("Embedding shape: ", x1.shape)

    s0 = Input(shape=(hidden_layer_size,), name='s0')
    c0 = Input(shape=(hidden_layer_size,), name='c0')

    attention_creator = AttentionModelCreator(Fx, Tx, Fy, Ty, hidden_layer_size,
                                              hidden_layer_size, e1=e1, e2=e2,
                                              activation="tanh")

    att1 = attention_creator.create_from_inputs(loss_func, layers_only=True,
                                                optimizer=optimizer,
                                                x=x1, s0=s0, c0=c0)

    att2 = attention_creator.create_from_inputs(loss_func, layers_only=True,
                                                optimizer=optimizer,
                                                x=x2, s0=s0, c0=c0)

    if len(att1) != 1:
        print("len(att1) = ", len(att1))
        raise AttributeError
    if len(att2) != 1:
        print("len(att2) = ", len(att2))
        raise AttributeError

    att1 = att1[0]
    att2 = att2[0]

    att1 = Reshape((embedding_size,))(att1)
    att2 = Reshape((embedding_size,))(att2)

    dot = Dot(-1, False)([att1, att2])
    x = Dense(1, activation='sigmoid')(dot)

    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=[x1_orig,x2_orig,s0,c0], outputs=x)
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
    def __init__(self, filepath, load_previous_model=True, word2vec_data=None, batch_size=512, word2vec_size=256,
                 max_len=500,
                 hidden_layer_size=128,
                 embedding_size=64,
                 e1=16, e2=8, lr=0.001,
                 loss_func='mean_squared_error',
                 callback=None):
        self.filepath = filepath
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.loss_func = loss_func
        self.word2vec_size = word2vec_size
        self.hidden_layer_size = hidden_layer_size
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
                word2vec_size, max_len,
                hidden_layer_size=hidden_layer_size,
                optimizer=Adam(lr=lr),
                e1=e1,
                e2=e2,
                embedding_size=embedding_size,
                word2vec_data=word2vec_data,
                loss_func=loss_func
            )

    def train(self, x1, x2, y, validation_data, epochs=1, shuffle=True, callbacks=None):
        m = y.shape[0]

        # define inputs
        s0 = np.zeros((m, self.hidden_layer_size))
        c0 = np.zeros((m, self.hidden_layer_size))

        #outputs = list(outputs.swapaxes(0, 1))

        # train
        self.model.fit([x1, x2, s0, c0], y, epochs=epochs, validation_data=validation_data,
                       batch_size=self.batch_size, shuffle=shuffle, callbacks=callbacks)
        if self.callback is not None:
            self.callback()

    def save(self):
        # save
        self.model.save(self.filepath)

    def load(self):
        self.model = load_rnn_encoding_model(self.filepath, lr=self.lr, loss_func=self.loss_func)



