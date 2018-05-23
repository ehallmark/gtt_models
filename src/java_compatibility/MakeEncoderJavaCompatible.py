from src.models.Word2VecCPCRnnEncodingModel import RnnEncoder
import pandas as pd
from src.models import Word2VecModel


def load_model(model_file):
    return RnnEncoder(model_file_64, load_previous_model=True,
                         loss_func='mean_squared_error')


def load_word2vec_index_maps():
    word2vec_index_file = '/home/ehallmark/Downloads/word2vec256_index.txt'
    word2vec_index = pd.read_csv(word2vec_index_file, sep=',')
    word_to_index_map = {}
    index_to_word_map = {}
    for row in word2vec_index:
        idx = int(row[1])
        word = str(row[0])
        if idx == 0:
            print("Found 0 index in word2vec... 0 index should be the mask index (i.e. not a word)."
                  + " Try rerunning IngestWord2VecToText java program.")
            raise ArithmeticError
        word_to_index_map[word] = idx
        index_to_word_map[idx] = word
    return word_to_index_map, index_to_word_map


def load_cpc2vec_index_maps():
    return Word2VecModel.build_dictionaries()


model_file_32 = '/home/ehallmark/data/python/w2v_cpc_rnn_model_keras32.h5'
model_file_64 = '/home/ehallmark/data/python/w2v_cpc_rnn_model_keras64.h5'
model_file_128 = '/home/ehallmark/data/python/w2v_cpc_rnn_model_keras128.h5'

word_idx_map, idx_word_map = load_word2vec_index_maps()
cpc_idx_map, idx_cpc_map = load_cpc2vec_index_maps()

encoder = load_model(model_file_64)
model = encoder.model

for layer in model.layers:
    print("Layer: "+layer.summary())


