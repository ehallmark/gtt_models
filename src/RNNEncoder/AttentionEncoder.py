import pandas as pd
import psycopg2
import numpy as np
from keras.callbacks import LearningRateScheduler
from src.models.RnnEncodingModel import RnnEncoder

data_csv_file = '/home/ehallmark/Downloads/attention_rnn_data.csv'
def get_data():
    data = pd.read_csv(data_csv_file, sep=',')

    return ((None, None), None), (None, None)


vocab_vector_file = '/home/ehallmark/Downloads/word2vec256_vectors.txt'
vocab_index_file = '/home/ehallmark/Downloads/word2vec256_index.txt'
model_file_32 = '/home/ehallmark/data/python/attention_rnn_model_keras32.h5'
model_file_64 = '/home/ehallmark/data/python/attention_rnn_model_keras64.h5'
model_file_128 = '/home/ehallmark/data/python/attention_rnn_model_keras128.h5'
vocab_size = 477909

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

    word_to_index = {}
    for row in pd.read_csv(vocab_index_file, sep=','):
        word_to_index[row[0]] = row[1]

    word2vec_data = np.loadtxt(vocab_vector_file)

    scheduler = LearningRateScheduler(lambda n: learning_rate/(max(1, n*5)))

    (data, data_val) = get_data()

    data, y = data
    x1, x2 = data
    data_val, y_val = data_val

    histories = []
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
        history = encoder.train(x1, x2, y, (data_val, y_val),
                                 epochs=epochs, shuffle=True, callbacks=[scheduler])
        print("History for model: ", history)
        histories.append(history)
        encoder.save()

    for i in range(len(histories)):
        print("History "+str(i), histories[i])

