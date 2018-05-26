from src.models.Word2VecCPCRnnEncodingModel import RnnEncoder
import pandas as pd
import psycopg2
import numpy as np
from src.models import Word2VecModel
from keras.models import Model
import re
from src.java_compatibility.BuildPatentEncodings import load_model, load_cpc2vec_index_maps, extract_cpc_model,\
    norm_across_rows
from keras.layers import Input


model_file_32 = '/home/ehallmark/data/python/w2v_cpc_rnn_model_keras32.h5'
model_file_64 = '/home/ehallmark/data/python/w2v_cpc_rnn_model_keras64.h5'
model_file_128 = '/home/ehallmark/data/python/w2v_cpc128_rnn_model_keras128.h5'


def cpc_sort(val):
    return len(val)


def encode_cpcs(model, cpc_to_idx, cpc_trees):
    all_cpcs = []
    all_weights = []
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
            cpcs.sort(key=cpc_sort)
            for j in range(len(cpcs)):
                cpc = cpcs[j]
                all_weights.append(np.exp(j))
                all_cpcs.append(cpc)

    x = np.array(all_cpcs).reshape((len(all_cpcs), 1, 1))
    y = model.predict(x) * np.array(all_weights).reshape(len(all_weights), 1)
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


def handle_batch(cnt, not_found):
    all_cpc_encoding = encode_cpcs(cpc_model, cpc_idx_map, cpc_batch)
    cpc_norm = norm_across_rows(all_cpc_encoding)
    all_cpc_encoding = all_cpc_encoding / np.where(cpc_norm != 0, cpc_norm, 1)[:, np.newaxis]
    all_vecs = all_cpc_encoding.tolist()
    all_norms = cpc_norm.tolist()
    data = []
    for i in range(len(all_vecs)):
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


if __name__ == '__main__':
    print("Starting to setup sql...")
    conn = psycopg2.connect("dbname='patentdb' user='postgres' host='localhost' password='password'")
    conn.autocommit = False
    conn2 = psycopg2.connect("dbname='patentdb' user='postgres' host='localhost' password='password'")
    conn2.autocommit = True
    seed_cursor = conn.cursor("stream")
    seed_cursor.itersize = 500
    seed_cursor.execute("""select code, tree from big_query_cpc_definition""")

    ingest_cursor = conn2.cursor()
    ingest_sql_pre = """insert into big_query_embedding_cpc (code,enc) values """
    ingest_sql_post = """ on conflict (code) do update set enc=excluded.enc"""

    print("Querying...")

    cpc_idx_map, idx_cpc_map = load_cpc2vec_index_maps()

    encoder = load_model(model_file_128)
    model = encoder.model
    model.summary()

    cpc_model = extract_cpc_model(model)

    # load model
    cnt = 0
    not_found = 0
    text_batch = []
    cpc_batch = []
    id_batch = []
    batch_size = 500
    for record in seed_cursor:
        id = record[0]
        cpc_tree = record[1]

        cpc_batch.append(cpc_tree)
        id_batch.append(id)
        if len(text_batch) >= batch_size:
            cnt, not_found = handle_batch(cnt, not_found)
            text_batch.clear()
            cpc_batch.clear()
            id_batch.clear()
    handle_batch(cnt, not_found)

    conn.close()
    conn2.close()

