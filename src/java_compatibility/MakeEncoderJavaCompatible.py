from src.models.Word2VecCPCRnnEncodingModel import RnnEncoder
import pandas as pd
import psycopg2
import numpy as np
from src.models import Word2VecModel
from keras.models import Model
import re
from keras.layers import Input

def load_model(model_file):
    return RnnEncoder(model_file, load_previous_model=True,
                         loss_func='mean_squared_error')


def load_word2vec_index_maps():
    word2vec_index_file = '/home/ehallmark/Downloads/word2vec256_index.txt'
    word2vec_index = np.loadtxt(word2vec_index_file, delimiter=',', dtype=(str,int))
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


def encode_text(model, word_to_idx, text):
    words = re.sub('[^a-z ]', ' ', text.lower()).split()
    words = [word_to_idx[word] for word in words if word in word_to_idx]
    if len(words) > 128:
        words = words[0:128]
    elif len(words) < 128:
        words = ([0] * (128 - len(words))) + words
    x = np.array(words).reshape((1, 128, 1))
    return model.predict(x).flatten()


def encode_cpcs(model, cpc_to_idx, cpcs):
    cpcs = [cpc_to_idx[cpc] for cpc in cpcs if cpc in cpc_to_idx]
    x = np.array(cpcs).reshape((len(cpcs), 1, 1))
    return model.predict(x).sum(0)


model_file_32 = '/home/ehallmark/data/python/w2v_cpc_rnn_model_keras32.h5'
model_file_64 = '/home/ehallmark/data/python/w2v_cpc_rnn_model_keras64.h5'
model_file_128 = '/home/ehallmark/data/python/w2v_cpc128_rnn_model_keras128.h5'

if __name__ == '__main__':
    print("Starting to setup sql...")
    conn = psycopg2.connect("dbname='patentdb' user='postgres' host='localhost' password='password'")
    conn2 = psycopg2.connect("dbname='patentdb' user='postgres' host='localhost' password='password'")
    seed_cursor = conn.cursor("stream")
    seed_cursor.itersize = 100
    seed_cursor.execute("""select p.family_id, p.abstract, tree from big_query_patent_english_abstract as p
       join big_query_cpc_tree as c on (p.publication_number_full=c.publication_number_full)""")

    ingest_cursor = conn2.cursor()
    ingest_sql = """insert into big_query_embedding_by_fam (family_id,enc) values (%s,%s) 
      on conflict (family_id) do update set enc=excluded.enc"""
    print("Querying...")

    word_idx_map, idx_word_map = load_word2vec_index_maps()
    cpc_idx_map, idx_cpc_map = load_cpc2vec_index_maps()
    print('Num words found: ', len(word_idx_map))

    encoder = load_model(model_file_128)
    model = encoder.model
    model.summary()

    cpc_input = Input((1, 1), dtype=np.int32)
    cpc_model = model.get_layer(name='embedding_2')(cpc_input)
    cpc_model = model.get_layer(name='flatten_1')(cpc_model)
    cpc_model = model.get_layer(name='dense_2')(cpc_model)
    cpc_model = model.get_layer(name='dense_3')(cpc_model)
    cpc_model = Model(inputs=cpc_input, outputs=cpc_model)
    cpc_model.compile(optimizer='adam', loss='mean_squared_error')
    print('cpc_model summary: ')
    cpc_model.summary()

    text_input = Input((None, 1), dtype=np.int32)
    text_model = model.get_layer(name='embedding_1')(text_input)
    text_model = model.get_layer(name='reshape_1')(text_model)
    text_model = model.get_layer(name='lstm_1')(text_model)
    text_model = model.get_layer(name='dense_1')(text_model)
    text_model = model.get_layer(name='dense_3')(text_model)
    text_model = Model(inputs=text_input, outputs=text_model)
    text_model.compile(optimizer='adam', loss='mean_squared_error')
    print('text_model summary: ')
    text_model.summary()

    # load model
    cnt = 0
    not_found = 0
    for record in seed_cursor:
        id = record[0]
        text = record[1]
        cpc_tree = record[2]

        cpc_encoding = encode_cpcs(cpc_model, cpc_idx_map, cpc_tree)
        text_encoding = encode_text(text_model, word_idx_map, text)
        if text_encoding is None and cpc_encoding is None:
            not_found = not_found + 1
        else:
            vec = None
            if text_encoding is not None and cpc_encoding is not None:
                vec = text_encoding / np.linalg.norm(text_encoding) + cpc_encoding / np.linalg.norm(cpc_encoding)
            elif text_encoding is not None:
                vec = text_encoding
            else:
                vec = cpc_encoding
            vec = vec / np.linalg.norm(vec)
            #print('ID: ', id)
            #print('Text: ', text)
            #print('CPC: ', cpc_tree)
            #print('Embedding', vec)
            ingest_cursor.execute(ingest_sql, (id, vec.tolist()))
        if cnt % 1000 == 999:
            print("Completed: ", cnt, ' Not found: ', not_found)
            conn2.commit()
        cnt = cnt + 1

    conn.close()
    conn2.close()

