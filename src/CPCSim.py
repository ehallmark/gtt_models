import keras as K
import numpy as np

n = 1000000             # number of examples
b_n = 512               # batch size
input_shape = (n, 1)
batch_shape = (b_n, 1)
e_n = 16                # embedding size
c_n = 200000            # num class codes
y_n = 1

# inputs must be the index of each class, NOT a one-hot vector
x1 = K.layers.Input(shape=input_shape, batch_shape=batch_shape, dtype=np.int, name="x1")
x2 = K.layers.Input(shape=input_shape, batch_shape=batch_shape, dtype=np.int, name="x2")

encoder = K.layers.Embedding(c_n, e_n, activation="tanh")

e1 = encoder(x1)
e2 = encoder(x2)

X = K.layers.merge([e1, e2], 'concat', -1)
X = K.layers.Dense(y_n, activation='sigmoid')(X)

model = K.models.Model(inputs=[x1, x2], outputs=X)


