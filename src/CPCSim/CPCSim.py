import keras as K
from keras.callbacks import Callback
import numpy as np
import psycopg2


class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data
        self.accuracies = []

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = model.evaluate(x, y, verbose=0)
        print("Previous accuracies: ", ", ".join(self.accuracies))
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
        self.accuracies.append(str(acc))



# where to save the model
model_file = 'cpc_sim_model64.h5'


# database
def train_iterator(conn):
    cur = conn.cursor()
    num_test = 15000
    cur.execute("""select id1,id2,freq from big_query_cpc_occurrence order by random()""")
    rows = cur.fetchall()
    x_train1 = np.zeros((len(rows)-num_test, 1), dtype=np.int32)
    x_train2 = np.zeros((len(rows)-num_test, 1), dtype=np.int32)
    y_train = np.zeros((len(rows)-num_test, 1), dtype=np.int32)
    x_test1 = np.zeros((num_test, 1), dtype=np.int32)
    x_test2 = np.zeros((num_test, 1), dtype=np.int32)
    y_test = np.zeros((num_test, 1), dtype=np.int32)
    x_val1 = np.zeros((num_test, 1), dtype=np.int32)
    x_val2 = np.zeros((num_test, 1), dtype=np.int32)
    y_val = np.zeros((num_test, 1), dtype=np.int32)
    r = 0
    for row in rows:
        if r < num_test:
            x_test1[r] = row[0]
            x_test2[r] = row[1]
            y_test[r] = row[2]
        elif r < num_test + num_test:
            x_val1[r - num_test] = row[0]
            x_val2[r - num_test] = row[1]
            y_val[r - num_test] = row[2]
        else:
            x_train1[r-2*num_test] = row[0]
            x_train2[r-2*num_test] = row[1]
            y_train[r-2*num_test] = row[2]
        r = r+1
    return [x_train1, x_train2], y_train, [x_val1, x_val2], y_val, [x_test1, x_test2], y_test, 259840


conn = psycopg2.connect("dbname='patentdb' user='postgres' host='localhost' password='password'")
X_train, Y_train, X_val, Y_val, X_test, Y_test, c_n = train_iterator(conn)

n = Y_train.shape[0]           # number of examples
b_n = 512               # batch size
input_shape = (1,)
e_n = 16
epochs = 10
good_enough_accuracy = 0.95

print("Num classes: ", c_n)
print("Num examples: ", n)

# inputs must be the index of each class, NOT a one-hot vector
x1 = K.layers.Input(shape=input_shape, dtype=np.int32, name="x1")
x2 = K.layers.Input(shape=input_shape, dtype=np.int32, name="x2")

encoder = K.layers.Embedding(c_n, e_n, input_length=1, trainable=True,
                             embeddings_initializer=K.initializers.RandomUniform(-0.1, 0.1),
                             embeddings_constraint=K.constraints.MaxNorm(1.),
                            # embeddings_regularizer=K.regularizers.l2(0.0001)
                             )
e1 = encoder(x1)
e2 = encoder(x2)

e1 = K.layers.Reshape((e_n,))(e1)
e2 = K.layers.Reshape((e_n,))(e2)

X = K.layers.Dot(1, False)([e1,e2])

X = K.layers.Dense(1, activation='sigmoid')(X)

model = K.models.Model(inputs=[x1, x2], outputs=X)

print("Model Summary", model.summary())

opt = K.optimizers.Adam(lr=0.001, decay=0.0001)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

callback = TestCallback((X_test, Y_test))
model.fit(X_train, Y_train, verbose=1, batch_size=b_n, epochs=epochs, shuffle=True, validation_data=(X_val, Y_val), callbacks=[callback])
# test

model.save(model_file, True, True)



