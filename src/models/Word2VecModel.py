from keras.models import Model
from keras.layers import Input, Dense, Reshape, Dot
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.initializers import RandomUniform
import pandas as pd
import psycopg2
import numpy as np
from keras.callbacks import LearningRateScheduler
import keras as k


def load_word2vec_model(model_file, lr=0.001,
                        loss_func='mean_squared_error'):
    print("Using previous model...")
    m = k.models.load_model(model_file, compile=False)
    m.compile(loss=loss_func, optimizer=Adam(lr=lr), metrics=['accuracy'])
    return m


def create_word2vec_model(embedding_size, vocab_size,
                          optimizer=Adam(lr=0.001), embedding_init=RandomUniform(-0.1, 0.1),
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
    m.compile(loss=loss_func, optimizer=optimizer, metrics=['accuracy'])
    return m


class Word2Vec:
    def __init__(self, filepath, load_previous_model=True, batch_size=512, embedding_size=64,
                 vocab_size=None, lr=0.001, decay=0.0,
                 loss_func='mean_squared_error', embeddings_initializer=RandomUniform(-0.1, 0.1),
                 callback=None):
        self.filepath = filepath
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.loss_func = loss_func
        self.lr = lr
        self.callback = callback
        self.embeddings_initializer = embeddings_initializer
        self.model = None
        if load_previous_model:
            try:
                self.load()
            except:
                print('Could not fine previous model... Creating new one now.')
        if self.model is None:
            optimizer = Adam(lr=lr, decay=decay)
            self.model = create_word2vec_model(embedding_size, vocab_size, optimizer=optimizer, loss_func=loss_func)

    def train(self, inputs, outputs, val_data, epochs=1, shuffle=True, callbacks=None):
        history = self.model.fit(inputs, outputs, verbose=1, batch_size=self.batch_size,
                       validation_data=val_data, epochs=epochs, shuffle=shuffle, callbacks=callbacks)
        if self.callback is not None:
            self.callback()
        return history

    def save(self):
        # save
        self.model.save(self.filepath)

    def load(self):
        self.model = load_word2vec_model(self.filepath, lr=self.lr, loss_func=self.loss_func)


model_file_32 = '/home/ehallmark/data/python/cpc_sim_model_keras_word2vec_32.h5'
model_file_64 = '/home/ehallmark/data/python/cpc_sim_model_keras_word2vec_64.h5'
model_file_128 = '/home/ehallmark/data/python/cpc_sim_model_keras_word2vec_128.h5'
def build_dictionaries():
    """Process raw inputs into a dataset."""
    dictionary = load_cpc_to_index_map()
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reversed_dictionary


# database
def load_cpc_data(randomize=True, sample_frac=1.0):
    train_file = '/home/ehallmark/Downloads/cpc_sim_data/data-0.csv'
    print("Reading dataset...")
    train_csv = pd.read_csv(train_file, header=0, sep=',', dtype=np.int32)
    if randomize:
        train_csv = train_csv.sample(frac=sample_frac)
    num_test = 50000
    x_train1, x_train2, y_train = train_csv.iloc[num_test:,0], train_csv.iloc[num_test:,1], train_csv.iloc[num_test:,2]
    print("Size of dataset: ", len(train_csv))
    x_val1, x_val2, y_val = train_csv.iloc[:num_test,0], train_csv.iloc[:num_test,1], train_csv.iloc[:num_test,2]
    print("Size of test set: ", len(x_val1))
    return ([x_train1, x_train2], y_train), ([x_val1, x_val2], y_val)


def load_cpc_to_index_map():
    cpc_to_index_map = {}
    conn = psycopg2.connect("dbname='patentdb' user='postgres' host='localhost' password='password'")
    cur = conn.cursor()
    cur.execute("""select code,id from big_query_cpc_occurrence_ids""")
    rows = cur.fetchall()
    for row in rows:
        cpc_to_index_map[row[0]] = row[1]
    conn.close()
    return cpc_to_index_map


if __name__ == "__main__":
    load_previous_model = False
    learning_rate = 0.001
    decay = 0.0001
    vocab_size = 259840
    batch_size = 512
    epochs = 1

    embedding_size_to_file_map = {
        32: model_file_32,
        64: model_file_64,
        128: model_file_128
    }

    scheduler = LearningRateScheduler(lambda n: learning_rate/(max(1, n*5)))

    (data, val_data) = load_cpc_data(randomize=True)
    dictionary, reverse_dictionary = build_dictionaries()
    ((word_target, word_context), labels) = data
    ((val_target, val_context), val_labels) = val_data

    histories = []
    for vector_dim, model_file in embedding_size_to_file_map.items():
        word2vec = Word2Vec(model_file, load_previous_model=load_previous_model, vocab_size=vocab_size,
                            batch_size=batch_size, loss_func='mean_squared_error',
                            embedding_size=vector_dim, lr=learning_rate)
        print("Starting to train model with embedding_size: ", vector_dim)
        history = word2vec.train([word_target, word_context], labels, ([val_target, val_context], val_labels),
                                 epochs=epochs, shuffle=True, callbacks=[scheduler])
        print("History for model: ", history)
        histories.append(history)
        word2vec.save()

    for i in range(len(histories)):
        print("History "+str(i), histories[i])

