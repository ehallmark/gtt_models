import pandas as pd
import psycopg2
import numpy as np
from keras.callbacks import LearningRateScheduler
from src.models.RnnEncodingModel import RnnEncoder


vocab_vector_file = '/home/ehallmark/Downloads/word2vec256_vectors.txt'
vocab_index_file = '/home/ehallmark/Downloads/word2vec256_index.txt'
model_file_32 = '/home/ehallmark/data/python/attention_rnn_model_keras32.h5'
model_file_64 = '/home/ehallmark/data/python/attention_rnn_model_keras64.h5'
model_file_128 = '/home/ehallmark/data/python/attention_rnn_model_keras128.h5'
vocab_size = 477909
def get_data():
    x1 = pd.read_csv('/home/ehallmark/Downloads/rnn_keras_x1.csv', sep=',')
    x2 = pd.read_csv('/home/ehallmark/Downloads/rnn_keras_x2.csv', sep=',')
    y = pd.read_csv('/home/ehallmark/Downloads/rnn_keras_y.csv', sep=',')

    seed = 1
    num_test = 20000
    x1_val = x1.sample(n=num_test, replace=False, random_state=seed)
    x2_val = x2.sample(n=num_test, replace=False, random_state=seed)
    y_val = y.sample(n=num_test, replace=False, random_state=seed)
    return ((x1, x2), y), ([x1_val, x2_val], y_val)


if __name__ == "__main__":
    load_previous_model = False
    learning_rate = 0.005
    decay = 0.0001
    batch_size = 512
    epochs = 1
    word2vec_size = 256

    embedding_size_to_file_map = {
        32: model_file_32,
        64: model_file_64,
        128: model_file_128
    }

    print('Loading word to index map...')
    word_to_index = {}
    rows = pd.read_csv(vocab_index_file, sep=',')
    for row in rows:
        word_to_index[row[0]] = row[1]

    print('Loading word2vec model...')
    word2vec_data = np.loadtxt(vocab_vector_file)

    scheduler = LearningRateScheduler(lambda n: learning_rate/(max(1, n*5)))

    print('Getting data...')
    (data, data_val) = get_data()

    data, y = data
    x1, x2 = data

    histories = []
    print('Training model...')
    for vector_dim, model_file in embedding_size_to_file_map.items():
        encoder = RnnEncoder(model_file, load_previous_model=load_previous_model,
                             hidden_layer_size=128, word2vec_size=word2vec_size,
                             batch_size=batch_size, loss_func='mean_squared_error',
                             embedding_size=vector_dim, lr=learning_rate,
                             max_len=10,
                             word2vec_data=word2vec_data,
                             word_to_index=word_to_index)
        print("Model Summary: ", encoder.model.summary())
        print("Starting to train model with embedding_size: ", vector_dim)
        history = encoder.train(x1, x2, y, data_val,
                                 epochs=epochs, shuffle=True, callbacks=[scheduler])
        print("History for model: ", history)
        histories.append(history)
        encoder.save()

    for i in range(len(histories)):
        print("History "+str(i), histories[i])

