from keras.models import Model
from keras.layers import Input, Dense, Reshape, Dot
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.initializers import RandomUniform
from keras.callbacks import LearningRateScheduler
import keras as k


def load_rnn_encoding_model(model_file, lr=0.001,
                            loss_func='mean_squared_error'):
    print("Using previous model...")
    m = k.models.load_model(model_file, compile=False)
    m.compile(loss=loss_func, optimizer=Adam(lr=lr), metrics=['accuracy'])
    return m


def create_rnn_encoding_model(embedding_size, word2vec_size, vocab_size,
                              lr=0.001, loss_func='mean_squared_error'):
    print("Creating new model...")
    # create some input variables
    inputs = []
    outputs = []

    # create the primary training model
    m = Model(input=inputs, output=outputs)
    m.compile(loss=loss_func, optimizer=Adam(lr=lr), metrics=['accuracy'])
    return m


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
        self.word2vec_size=word2vec_size
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

    def train(self, inputs, outputs, val_data, epochs=1, shuffle=True, callbacks=[]):
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



