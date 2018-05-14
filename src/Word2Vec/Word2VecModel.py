from keras.models import Model
from keras.layers import Input, Dense, Reshape, Dot
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.initializers import RandomUniform
import keras as k


def load_word2vec_model(model_file, lr=0.001,
                        loss_func='mean_squared_error'):
    print("Using previous model...")
    m = k.models.load_model(model_file, compile=False)
    m.compile(loss=loss_func, optimizer=Adam(lr=lr), metrics=['accuracy'])
    return m


def create_word2vec_model(embedding_size, vocab_size,
                          lr=0.001, embedding_init=RandomUniform(-0.1, 0.1),
                          loss_func='mean_squared_error'):
    print("Creating new model...")
    # create some input variables
    input_target = Input((1,))
    input_context = Input((1,))

    embedding = Embedding(vocab_size, embedding_size, input_length=1, name='embedding',
                          embeddings_initializer=embedding_init,
                          )

    target = embedding(input_target)
    target = Reshape((embedding_size, 1))(target)
    context = embedding(input_context)
    context = Reshape((embedding_size, 1))(context)

    # now perform the dot product operation to get a similarity measure
    dot_product = Dot(1, False)([target, context])
    dot_product = Reshape((1,))(dot_product)
    # add the sigmoid output layer
    output = Dense(1, activation='sigmoid')(dot_product)
    # create the primary training model
    m = Model(input=[input_target, input_context], output=output)
    m.compile(loss=loss_func, optimizer=Adam(lr=lr), metrics=['accuracy'])
    return m


class Word2Vec:
    def __init__(self, filepath, load_previous_model=True, batch_size=512, embedding_size=64,
                 vocab_size=None, lr=0.001,
                 loss_func='mean_squared_error', embeddings_initializer=RandomUniform(-0.1, 0.1),
                 callback=None):
        self.filepath = filepath
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.loss_func = loss_func
        self.lr = lr
        self.callback = callback
        self.embeddings_initializer = embeddings_initializer

        if load_previous_model:
            try:
                self.load()
            except:
                print('Could not fine previous model... Creating new one now.')
        if self.model is None:
            self.model = create_word2vec_model(embedding_size, vocab_size, lr=lr, loss_func=loss_func)

    def train(self, inputs, outputs, val_data, epochs=1, shuffle=True):
        for cnt in range(epochs):
            self.model.fit(inputs, outputs, verbose=1, batch_size=self.batch_size,
                           validation_data=val_data, shuffle=shuffle)
            if self.callback is not None:
                self.callback()

    def save(self):
        # save
        self.model.save(self.filepath)

    def load(self):
        self.model = load_word2vec_model(self.filepath, lr=self.lr, loss_func=self.loss_func)



