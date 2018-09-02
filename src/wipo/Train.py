from sqlalchemy import create_engine
import pandas as pd
conn = create_engine("postgresql://localhost/patentdb?user=postgres&password=password")
from sklearn import preprocessing
import numpy as np
from keras.layers import Dense, Reshape, Bidirectional, Embedding, Add, Multiply, Concatenate, Lambda, Input, BatchNormalization, Dropout
from keras.models import Model
from keras.optimizers import Adam
import sklearn.metrics as metrics
import numpy.random as random
random.seed(235)
num_tests = 25000
data_file = 'join_data.h5'
cpc_definition_file = 'cpc_definition.h5'
use_cache = False

def random_select(arr):
    if len(arr) == 0:
        return None
    r = random.randint(0, len(arr))
    return arr[r]

def test_model(model, x, y):
    y_pred = model.predict(x)
    return metrics.log_loss(y, y_pred)


print('Loading data')
if use_cache:
    data = pd.read_hdf(data_file, 'data')
    cpc_definitions_sql = pd.read_hdf(cpc_definition_file, 'definition')
else:
    cpc_definitions_sql = pd.read_sql('select code from big_query_cpc_definition where not code like \'%%/%%\'', conn)
    cpc_definitions_sql.to_hdf(cpc_definition_file, 'definition', mode='w')
    data = pd.read_sql('select c.publication_number_full, wipo_technology, tree from big_query_wipo_by_pub_flat as w join big_query_cpc_tree as c on (c.publication_number_full=w.publication_number_full)', conn)
    data['tree'] = [random_select([d for d in dat if not '/' in d]) for dat in data['tree']]
    data.to_hdf(data_file, 'data', mode='w')
    print('Loading cpc definitions')


print('Done')

print('Data size: ', data.shape)

cpc_encoder = preprocessing.LabelEncoder()
cpc_encoder.fit(list(cpc_definitions_sql['code'].iloc[:]))

wipo_encoder = preprocessing.LabelEncoder()
wipo_encoder.fit(list(data['wipo_technology'].iloc[:]))

print("Num wipo classes: ", len(wipo_encoder.classes_))
print("Num cpc classes: ", len(cpc_encoder.classes_))

data = data.sample(frac=1).reset_index(drop=True)  # shuffle
test_data = data.iloc[-num_tests:]
data = data.iloc[0:-num_tests]

print('Building wipo dataset')
wipo = wipo_encoder.transform(data['wipo_technology'].iloc[:])
wipo_test = wipo_encoder.transform(test_data['wipo_technology'].iloc[:])
wipo_encod = np.zeros([len(wipo), len(wipo_encoder.classes_)])
wipo_encod_test = np.zeros([len(wipo_test), len(wipo_encoder.classes_)])
for i in range(data.shape[0]):
    wipo_encod[i, wipo[i]] = 1
for i in range(test_data.shape[0]):
    wipo_encod_test[i, wipo_test[i]] = 1

print('Building cpc dataset')
cpc = cpc_encoder.transform(data['tree'].iloc[:])
cpc_test = cpc_encoder.transform(test_data['tree'].iloc[:])

test_data = (cpc_test, wipo_encod_test)

print('Final wipo shape', wipo_encod.shape)
print('Final cpc shape', wipo_encod.shape)

model_file = 'wipo_prediction_model.h5'
hidden_units = 512
num_layers = 4
dropout = 0.5
batch_size = 256

X = Input((len(cpc_encoder.classes_),))

model = X
embedding = Embedding(len(cpc_encoder.classes_), hidden_units)
model = embedding(model)
for i in range(num_layers):
    model = Dense(hidden_units, activation='tanh')(model)
    if dropout > 0:
        model = Dropout(dropout)(model)

model = Dense(len(wipo_encoder.classes_), activation='softmax')(model)

model = Model(inputs=X, outputs=model)
model.compile(optimizer=Adam(lr=0.0005, decay=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
avg_error = test_model(model, test_data[0], test_data[1])
print("Starting model score: ", avg_error)
prev_error = avg_error
best_error = avg_error
errors = []
errors.append(avg_error)
for i in range(30):
    model.fit(cpc, wipo_encod, batch_size=batch_size, initial_epoch=i, epochs=i + 1, validation_data=test_data,
              shuffle=True)
    avg_error = test_model(model, test_data[0], test_data[1])
    print('Average error: ', avg_error)
    if best_error is None or best_error > avg_error:
        best_error = avg_error
        # save
        model.save(model_file)
        print('Saved.')
    prev_error = avg_error
    errors.append(prev_error)

print(model.summary())
print('Most recent model error: ', prev_error)
print('Best model error: ', best_error)
print("Error history: ", errors)
