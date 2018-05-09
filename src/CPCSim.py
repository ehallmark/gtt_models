import keras as K
import numpy as np
import psycopg2


# database
def train_iterator(conn):
    try:
        c_n = conn.cursor().execute("""select count(*) from big_query_cpc_occurrence_ids""")
        c_n = c_n[0]
        cur = conn.cursor()
        cur.execute("""select id1,id2,freq from big_query_cpc_occurrence order by random() tablesample system (1) """)
        rows = cur.fetchall()
    except:
        print("I am unable to connect to the database")
        exit(1)
        rows = []
        c_n = 0
    x_train1 = np.array(len(rows), 1)
    x_train2 = np.array(len(rows), 1)
    y_train = np.zeros(len(rows), 1)
    r = 0
    for row in rows:
        x_train1[r] = row[1]
        x_train2[r] = row[2]
        y_train[r] = row[3]
        r = r+1
    return (x_train1, x_train2), y_train, c_n


conn = psycopg2.connect("dbname='patentdb' user='postgres' host='localhost' password='password'")
X_train, Y_train, c_n = train_iterator(conn)

n = Y_train.shape[0]           # number of examples
b_n = 512               # batch size
input_shape = (n, 1)
e_n = 16                # embedding size
y_n = 1

print("Num classes: ", c_n)
print("Num examples: ", n)


# inputs must be the index of each class, NOT a one-hot vector
x1 = K.layers.Input(shape=input_shape, dtype=np.int32, name="x1")
x2 = K.layers.Input(shape=input_shape, dtype=np.int32, name="x2")

encoder = K.layers.Embedding(c_n, e_n, activation="tanh")

e1 = encoder(x1)
e2 = encoder(x2)

X = K.layers.merge([e1, e2], 'dot', -1, -1)

model = K.models.Model(inputs=[x1, x2], outputs=X)

print("Model Summary", model.summary())

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=b_n, epochs=10, shuffle=True)

