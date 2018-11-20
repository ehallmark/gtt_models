from sqlalchemy import create_engine
import pandas as pd
conn = create_engine("postgresql://localhost/patentdb?user=postgres&password=password")
from sklearn import preprocessing
import numpy as np
from keras.layers import Dense, Reshape, Bidirectional, Flatten, Embedding, Add, Multiply, Concatenate, Lambda, Input, BatchNormalization, Dropout
from keras.models import Model
from keras.optimizers import Adam
import keras as K
import sklearn.metrics as metrics
import numpy.random as random
import psycopg2
random.seed(235)
num_tests = 25000
data_file = 'join_data.h5'
cpc_definition_file = 'cpc_definition.h5'


print('Loading data')

wipo_sql = pd.read_hdf(data_file, 'data')
cpc_definitions_sql = pd.read_hdf(cpc_definition_file, 'definition')
data = pd.read_sql('select c.publication_number_full, tree from big_query_wipo_prediction as w full outer join big_query_cpc_tree as c on (c.publication_number_full=w.publication_number_full) where w.publication_number_full is null', conn)
data['tree'] = [[d for d in dat if len(d) < 8] for dat in data['tree']]

print('Done')

print('Data size: ', data.shape)

cpc_encoder = preprocessing.LabelEncoder()
cpc_encoder.fit(list(cpc_definitions_sql['code'].iloc[:]))

wipo_encoder = preprocessing.LabelEncoder()
wipo_encoder.fit(list(wipo_sql['wipo_technology'].iloc[:]))

print("Num wipo classes: ", len(wipo_encoder.classes_))
print("Num cpc classes: ", len(cpc_encoder.classes_))

print('Building cpc dataset')


def batch_generator(start_idx, batch_size=10000):
    cpc_encod = np.zeros([min(batch_size,data.shape[0]-start_idx), len(cpc_encoder.classes_)])
    idx = 0
    print('Start ', start_idx)
    for i in range(start_idx, min(start_idx+batch_size,data.shape[0])):
        cpc = cpc_encoder.transform(data['tree'].iloc[i])
        for j in range(len(cpc)):
            cpc_encod[idx, cpc[j]] = 1
        idx += 1
    return cpc_encod


model_file = 'wipo_prediction_model.h5'
hidden_units = 512
num_layers = 3
dropout = 0
batch_size = 10000

model = K.models.load_model(model_file)
model.compile(optimizer=Adam(lr=0.001, decay=0.1), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

conn = psycopg2.connect("postgresql://localhost/patentdb?user=postgres&password=password")
cursor = conn.cursor()

i = 0
while i < data.shape[0]:
    batch = batch_generator(i, batch_size)
    predictions = model.predict(batch, batch_size=256)
    print('Predictions shape: ', predictions.shape)
    predictions = np.argmax(predictions, 1)
    predictions = wipo_encoder.inverse_transform(predictions)
    for j in range(predictions.shape[0]):
        prediction = predictions[j]
        label = data['publication_number_full'].iloc[i+j]
        # print('Prediction for ' + label, prediction)
        insert_str = '''
            insert into big_query_wipo_prediction (publication_number_full, wipo_technology) 
            values (\'{{LABEL}}\', \'{{VALUE}}\') 
            on conflict (publication_number_full) do update set wipo_technology=excluded.wipo_technology
            '''.replace("{{LABEL}}", label).replace("{{VALUE}}", prediction)
        cursor.execute(insert_str)
        if j == predictions.shape[0]-1:
            print('Sample ', label+':', prediction)

    i += batch_size
    conn.commit()


conn.commit()
cursor.close()
conn.close()

