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
    word2vec_index_file = '/home/ehallmark/data/python/word2vec256_index.txt'
    word2vec_index = np.array(pd.read_csv(word2vec_index_file, delimiter=',', header=None))
    print(word2vec_index[0:10])
    word_to_index_map = {}
    index_to_word_map = {}
    for row in word2vec_index:
        idx = int(row[1])
        word = str(row[0])
        print('word: ', word, ' idx: ',idx)
        if idx == 0:
            print("Found 0 index in word2vec... 0 index should be the mask index (i.e. not a word)."
                  + " Try rerunning IngestWord2VecToText java program.")
            raise ArithmeticError
        word_to_index_map[word] = idx
        index_to_word_map[idx] = word
    return word_to_index_map, index_to_word_map


def load_cpc2vec_index_maps():
    return Word2VecModel.build_dictionaries()


def encode_text(model, word_to_idx, texts):
    all_vectors = []
    all_words = []
    invalid = set([])
    for i in range(len(texts)):
        text = texts[i]
        if text is not None:
            words = re.sub('[^a-z ]', ' ', text.lower()).split()
            words = [word_to_idx[word] for word in words if word in word_to_idx]
        else:
            words = []
        if len(words) < 3:
            invalid.add(i)
            words = [0] * 128
        else:
            if len(words) > 128:
                words = words[0:128]
            elif len(words) < 128:
                words = ([0] * (128 - len(words))) + words
        all_words.append(words)
    x = np.array(all_words).reshape((len(all_words), 128, 1))
    y = model.predict(x)
    for i in range(y.shape[0]):
        if i in invalid:
            all_vectors.append([0] * 128)
        else:
            all_vectors.append(y[i, :].flatten())
    return np.vstack(all_vectors)


def encode_cpcs(model, cpc_to_idx, cpc_trees):
    all_cpcs = []
    intervals = []
    invalid = set()
    for i in range(len(cpc_trees)):
        cpcs = cpc_trees[i]
        if cpcs is None:
            invalid.add(i)
            intervals.append(0)
        else:
            cpcs = [cpc_to_idx[cpc] for cpc in cpcs if cpc in cpc_to_idx]
            intervals.append(len(cpcs))
            for cpc in cpcs:
                all_cpcs.append(cpc)

    x = np.array(all_cpcs).reshape((len(all_cpcs), 1, 1))
    y = model.predict(x)
    vecs = []
    idx = 0
    for i in range(len(intervals)):
        interval = intervals[i]
        if i in invalid:
            vecs.append([0] * 128)
        else:
            vecs.append(y[idx:idx+interval, :].sum(0))
        idx += interval
    return np.vstack(vecs)


def extract_cpc_model(model):
    cpc_input = Input((1, 1), dtype=np.int32)
    cpc_model = model.get_layer(name='embedding_2')(cpc_input)
    cpc_model = model.get_layer(name='flatten_1')(cpc_model)
    cpc_model = model.get_layer(name='dense_2')(cpc_model)
    cpc_model = model.get_layer(name='dense_3')(cpc_model)
    cpc_model = Model(inputs=cpc_input, outputs=cpc_model)
    cpc_model.compile(optimizer='adam', loss='mean_squared_error')
    print('cpc_model summary: ')
    cpc_model.summary()
    return cpc_model


def extract_text_model(model):
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
    return text_model


def handle_batch(cnt, not_found):
    all_cpc_encoding = encode_cpcs(cpc_model, cpc_idx_map, cpc_batch)
    all_text_encoding = encode_text(text_model, word_idx_map, text_batch)
    cpc_norm = norm_across_rows(all_cpc_encoding)
    all_cpc_encoding = all_cpc_encoding / np.where(cpc_norm != 0, cpc_norm, 1)[:, np.newaxis]
    text_norm = norm_across_rows(all_text_encoding)
    all_text_encoding = all_text_encoding / np.where(text_norm != 0, text_norm, 1)[:, np.newaxis]

    all_vecs = all_text_encoding + all_cpc_encoding
    all_norms = norm_across_rows(all_vecs)
    all_vecs = all_vecs / all_norms[:, np.newaxis]
    all_vecs = all_vecs.tolist()
    all_norms = all_norms.tolist()
    data = []
    for i in range(len(all_cpc_encoding)):
        if all_norms[i] == 0:
            not_found = not_found + 1
        else:
            vec = all_vecs[i]
            data.append((id_batch[i], vec))
        if cnt % 1000 == 999:
            print("Completed: ", cnt, ' Not found: ', not_found)

        cnt = cnt + 1

    dataText = ','.join(ingest_cursor.mogrify('(%s,%s)', row).decode("utf-8") for row in data)
    ingest_cursor.execute(ingest_sql_pre + dataText + ingest_sql_post)
    return cnt, not_found


def norm_across_rows(matrix):
    return np.sum(matrix ** 2, axis=-1) ** (1./2)


model_file_32 = '/home/ehallmark/data/python/w2v_cpc_rnn_model_keras32.h5'
model_file_64 = '/home/ehallmark/data/python/w2v_cpc_rnn_model_keras64.h5'
model_file_128 = '/home/ehallmark/data/python/w2v_cpc128_rnn_model_keras128.h5'

if __name__ == '__main__':
    print("Starting to setup sql...")
    redo_embeddings = True   # determines whether or not to redo embeddings
    conn = psycopg2.connect("dbname='patentdb' user='postgres' host='localhost' password='password'")
    conn.autocommit = False
    conn2 = psycopg2.connect("dbname='patentdb' user='postgres' host='localhost' password='password'")
    conn2.autocommit = True
    seed_cursor = conn.cursor("stream")
    seed_cursor.itersize = 500
    seed_sql = """select coalesce(p.family_id,c.family_id), p.abstract, c.tree from big_query_patent_english_abstract as p
       full outer join big_query_cpc_tree_by_fam as c on (p.family_id=c.family_id)
       where p.family_id != '-1' and c.family_id != '-1' """
    # seed_sql = """select c.family_id, p.abstract, c.tree from big_query_patent_english_abstract as p
    #       full outer join big_query_cpc_tree_by_fam as c on (p.family_id=c.family_id)
    #       where p.family_id is null and c.family_id != '-1' """ # ONLY TO UPDATE MISSING VECS WITH ONLY CPCS
    if not redo_embeddings:
        seed_sql = seed_sql + ''' and not coalesce(p.family_id,c.family_id) 
                in (select family_id from big_query_embedding_by_fam)'''

    seed_cursor.execute(seed_sql)

    ingest_cursor = conn2.cursor()
    ingest_sql_pre = """insert into big_query_embedding_by_fam (family_id,enc) values """
    ingest_sql_post = """ on conflict (family_id) do update set enc=excluded.enc"""

    print("Querying...")

    word_idx_map, idx_word_map = load_word2vec_index_maps()
    cpc_idx_map, idx_cpc_map = load_cpc2vec_index_maps()
    print('Num words found: ', len(word_idx_map.keys()))

    encoder = load_model(model_file_128)
    model = encoder.model
    model.summary()

    cpc_model = extract_cpc_model(model)
    text_model = extract_text_model(model)

    # load model
    cnt = 0
    not_found = 0
    text_batch = []
    cpc_batch = []
    id_batch = []
    batch_size = 500
    for record in seed_cursor:
        id = record[0]
        text = record[1]
        cpc_tree = record[2]

        text_batch.append(text)
        cpc_batch.append(cpc_tree)
        id_batch.append(id)
        if len(text_batch) >= batch_size:
            cnt, not_found = handle_batch(cnt, not_found)
            text_batch.clear()
            cpc_batch.clear()
            id_batch.clear()
    if len(text_batch) > 0:
        handle_batch(cnt, not_found)

    conn.close()
    conn2.close()

