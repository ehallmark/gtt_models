import pandas as pd
import psycopg2
import numpy as np
from src.Word2Vec.Word2VecModel import Word2Vec

model_file = '/home/ehallmark/data/python/cpc_similarity_model_keras_word2vec_64.h5'

def build_dictionaries():
    """Process raw inputs into a dataset."""
    dictionary = load_cpc_to_index_map()
    count = []
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reversed_dictionary


# database
def load_cpc_data():
    train_file = '/home/ehallmark/Downloads/cpc_sim_data/data-0.csv'
    print("Reading dataset...")
    train_csv = pd.read_csv(train_file, header=0, sep=',', dtype=np.int32)
    train_csv = train_csv.sample(frac=1)
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


load_previous_model = True
learning_rate = 0.0001
vocab_size = 259840
vector_dim = 64
batch_size = 512
epochs = 1

word2vec = Word2Vec(model_file, load_previous_model=load_previous_model, batch_size=batch_size,
                    embedding_size=vector_dim, lr=learning_rate, loss_func='mean_squared_error')

print("Model compiled.")

valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

(data, val_data) = load_cpc_data()
dictionary, reverse_dictionary = build_dictionaries()
((word_target, word_context), labels) = data
((val_target, val_context), val_labels) = val_data

word2vec.train([word_target, word_context], labels, ([val_target, val_context], val_labels),
               epochs=epochs, shuffle=True)
# save
word2vec.save()
