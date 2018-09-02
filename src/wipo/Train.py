from sqlalchemy import create_engine
import pandas as pd
conn = create_engine("postgresql://localhost/patentdb?user=postgres&password=password")
from sklearn import preprocessing
import numpy as np
from keras.layers import Dense, Reshape, Bidirectional, Add, Multiply, Concatenate, Lambda, Input, BatchNormalization, Dropout
from keras.models import Model
from keras.optimizers import Adam
import sklearn.metrics as metrics

num_tests = 25000
data_file = 'join_data.h5'
cpc_definition_file = 'cpc_definition.h5'
use_cache = True


def test_model(model, x, y):
    y_pred = model.predict(x)
    return metrics.log_loss(y, y_pred)


print('Loading data')
if use_cache:
    data = pd.read_hdf(data_file, 'data')
    cpc_definitions_sql = pd.read_hdf(cpc_definition_file, 'definition')
else:
    data = pd.read_sql('select c.publication_number_full, wipo_technology, code from big_query_wipo_by_pub_flat as w join big_query_cpc_tree as c on (c.publication_number_full=w.publication_number_full)', conn)
    data.to_hdf(data_file, 'data', mode='w')
    print('Loading cpc definitions')
    cpc_definitions_sql = pd.read_sql('select code from big_query_cpc_definition where level < 8', conn)
    cpc_definitions_sql.to_hdf(cpc_definition_file, 'definition', mode='w')

print('Done')

print('Data size: ', data.shape)

cpc_encoder = preprocessing.LabelEncoder()
cpc_encoder.fit(list(cpc_definitions_sql['code'].iloc[:]))

wipo_encoder = preprocessing.LabelEncoder()
wipo_encoder.fit(list(data['wipo_technology'].iloc[:]))

data = data.sample(frac=1).reset_index(drop=True)  # shuffle
test_data = data.iloc[-num_tests:]
data = data.iloc[0:-num_tests]

print('Building wipo dataset')
wipo = wipo_encoder.transform(data['wipo_technology'].iloc[:])
wipo_test = wipo_encoder.transform(test_data['wipo_technology'].iloc[:])
wipo_encod = np.zeros([data.shape[0],len(wipo_encoder.classes_)])
wipo_encod_test = np.zeros([test_data.shape[0],len(wipo_encoder.classes_)])
for i in range(data.shape[0]):
    wipo_encod[i, wipo[i]] = 1.0
for i in range(test_data.shape[0]):
    wipo_encod_test[i, wipo_test[i]] = 1.0

print('Building cpc dataset')
cpc_encod = np.zeros([data.shape[0],len(cpc_encoder.classes_)])
cpc_encod_test = np.zeros([test_data.shape[0],len(cpc_encoder.classes_)])
for i in range(data.shape[0]):
    row = data.iloc[i]['tree']
    row = [r for r in row if r in cpc_encoder.classes_]
    if len(row) > 0:
        row = cpc_encoder.transform(row)
        for j in range(len(row)):
            cpc_encod[i, row[j]] = 1.0
for i in range(test_data.shape[0]):
    row = test_data.iloc[i]['tree']
    row = [r for r in row if r in cpc_encoder.classes_]
    if len(row) > 0:
        row = cpc_encoder.transform(row)
        for j in range(len(row)):
            cpc_encod_test[i, row[j]] = 1.0


test_data = (cpc_encod_test, wipo_encod_test)

print('Final wipo shape', wipo_encod.shape)
print('Final cpc shape', wipo_encod.shape)

model_file = 'wipo_prediction_model.h5'
hidden_units = 256
num_layers = 3
dropout = 0.5
batch_size = 256

X = Input((cpc_encoder.classes_,))

model = X
for i in range(num_layers):
    model = Dense(hidden_units, activation='tanh')(model)
    if dropout > 0:
        model = Dropout(dropout)(model)

model = Dense(wipo_encoder.classes_, activation='softmax')(model)

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
    model.fit(cpc_encod, wipo_encod, batch_size=batch_size, initial_epoch=i, epochs=i + 1, validation_data=test_data,
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
