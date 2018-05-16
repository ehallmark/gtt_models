import pandas as pd
import psycopg2
import numpy as np
from keras.callbacks import LearningRateScheduler
from src.models.RnnEncodingModel import RnnEncoder


model_file_32 = '/home/ehallmark/data/python/attention_rnn_model_keras32.h5'
model_file_64 = '/home/ehallmark/data/python/attention_rnn_model_keras64.h5'
model_file_128 = '/home/ehallmark/data/python/attention_rnn_model_keras128.h5'

load_previous_model = False
learning_rate = 0.005
decay = 0.0001
vocab_size = 259840
batch_size = 512
epochs = 1
word2vec_size = 256

embedding_size_to_file_map = {
    32: model_file_32,
    64: model_file_64,
    128: model_file_128
}

word_to_index = {
    'apple': 0,
    'orange': 1,
    'strawberry': 2,
    'cucumber': 3
}

word_to_vec_map = {}

for word, index in word_to_index.items():
    word_to_vec_map[word] = np.random.uniform(-1,1, (word2vec_size,))

scheduler = LearningRateScheduler(lambda n: learning_rate/(max(1, n*5)))

(data, data_val) = ((None, None), None), (None,None)

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
                         word_to_vec_map=word_to_vec_map,
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

