import keras as K
import numpy as np
import psycopg2


model = K.models.load_model('cpc_sim_model64.h5')


weights = model.get_layer(index=0).get_weights()
print("Weights shape: ",weights.shape)



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


cpc_to_index_map = load_cpc_to_index_map()
def get_embedding(cpc, weights):
    idx = cpc_to_index_map[cpc]
    if idx is not None and idx >= 0:
        return weights[:][idx]
    else:
        return None

