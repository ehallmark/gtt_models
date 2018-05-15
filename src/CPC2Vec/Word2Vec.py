import pandas as pd
import psycopg2
import numpy as np
from keras.callbacks import LearningRateScheduler
from src.models.Word2VecModel import Word2Vec


model_file_32 = '/home/ehallmark/data/python/cpc_sim_model_keras_word2vec_32.h5'
model_file_64 = '/home/ehallmark/data/python/cpc_sim_model_keras_word2vec_64.h5'
model_file_128 = '/home/ehallmark/data/python/cpc_sim_model_keras_word2vec_128.h5'
def build_dictionaries():
    """Process raw inputs into a dataset."""
    dictionary = load_cpc_to_index_map()
    count = []
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reversed_dictionary


# database
def load_cpc_data(randomize=True):
    train_file = '/home/ehallmark/Downloads/cpc_sim_data/data-0.csv'
    print("Reading dataset...")
    train_csv = pd.read_csv(train_file, header=0, sep=',', dtype=np.int32)
    if randomize:
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


load_previous_model = False
learning_rate = 0.01
decay = 0.0001
vocab_size = 259840
batch_size = 512
epochs = 200
sampling_per_epoch = 5000000

embedding_size_to_file_map = {
    32: model_file_32,
    64: model_file_64,
    128: model_file_128
}

# scheduler = LearningRateScheduler(lambda n: learning_rate/(max(1, n*5)))

(data, val_data) = load_cpc_data(randomize=False)
dictionary, reverse_dictionary = build_dictionaries()
((word_target, word_context), labels) = data
((val_target, val_context), val_labels) = val_data


models = []
for vector_dim, model_file in embedding_size_to_file_map.items():
    word2vec = Word2Vec(model_file, load_previous_model=load_previous_model, vocab_size=vocab_size,
                        batch_size=batch_size, loss_func='mean_squared_error',
                        embedding_size=vector_dim, decay=decay, lr=learning_rate)
    models.append(word2vec)

print("Models compiled.")
for i in range(epochs):
    word_target_sample = word_target.sample(n=sampling_per_epoch, replace=True, seed=i)
    word_context_sample = word_context.sample(n=sampling_per_epoch, replace=True, seed=i)
    labels_sample = labels.sample(n=sampling_per_epoch, replace=True, seed=i)
    n = 0
    for word2vec in models:
        history = word2vec.train([word_target_sample, word_context_sample], labels_sample, ([val_target, val_context], val_labels),
                             epochs=epochs, shuffle=True, callbacks=None)
        print("History for model "+str(n)+": ", history)
        n = n + 1

print("Saving models...")
for word2vec in models:
    # save
    word2vec.save()
