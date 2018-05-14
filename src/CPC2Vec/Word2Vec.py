from keras.models import Model
from keras.layers import Input, Dense, Reshape, Dot, Dropout
from keras.layers.embeddings import Embedding
import pandas as pd
from keras.optimizers import RMSprop, Adam
from keras.initializers import RandomUniform
import psycopg2
import numpy as np


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


vocab_size = 259840

vector_dim = 64
batch_size = 1024
epochs = 1

valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

(data, val_data) = load_cpc_data()
dictionary, reverse_dictionary = build_dictionaries()
((word_target, word_context), labels) = data
((val_target, val_context), val_labels) = val_data
# create some input variables
input_target = Input((1,))
input_context = Input((1,))

embedding = Embedding(vocab_size, vector_dim, input_length=1, name='embedding',
                      embeddings_initializer=RandomUniform(-0.1, 0.1),
                      )
target = embedding(input_target)
target = Reshape((vector_dim, 1))(target)
context = embedding(input_context)
context = Reshape((vector_dim, 1))(context)
# setup a cosine similarity operation which will be output in a secondary model
similarity = Dot(0, True)([target, context])

# now perform the dot product operation to get a similarity measure
dot_product = Dot(1, False)([target, context])
dot_product = Reshape((1,))(dot_product)
# add the sigmoid output layer
output = Dense(1, activation='sigmoid')(dot_product)
# create the primary training model
model = Model(input=[input_target, input_context], output=output)
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001), metrics=['accuracy'])

# create a secondary validation model to run our similarity checks during training
validation_model = Model(input=[input_target, input_context], output=similarity)


class SimilarityCallback:
    def run_sim(self):
        for i in range(valid_size):
            valid_word = reverse_dictionary[valid_examples[i]]
            top_k = 8  # number of nearest neighbors
            sim = self._get_sim(valid_examples[i])
            nearest = (-sim).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % valid_word
            for k in range(top_k):
                close_word = reverse_dictionary[nearest[k]]
                log_str = '%s %s,' % (log_str, close_word)
            print(log_str)

    @staticmethod
    def _get_sim(valid_word_idx):
        sim = np.zeros((vocab_size,))
        in_arr1 = np.zeros((1,))
        in_arr2 = np.zeros((1,))
        in_arr1[0,] = valid_word_idx
        for i in range(vocab_size):
            in_arr2[0,] = i
            out = validation_model.predict_on_batch([in_arr1, in_arr2])
            sim[i] = out
        return sim


sim_cb = SimilarityCallback()

for cnt in range(epochs):
    loss = model.fit([word_target, word_context], labels, verbose=1, batch_size=batch_size,
                     validation_data=([val_target, val_context], val_labels), shuffle=True)
    print("Epoch {}, loss={}".format(cnt, loss))

# save
model.save(model_file)
# sample
sim_cb.run_sim()
