import keras as K
import numpy as np


input_shape = None
batch_shape = None
e_n = 16                # embedding size
c_n = input_shape[1]    # num class codes
y_n = 1

x1 = K.layers.Input(shape=input_shape, batch_shape=batch_shape, name="x1", sparse=True)
x2 = K.layers.Input(shape=input_shape, batch_shape=batch_shape, name="x2", sparse=True)

encoder = K.layers.Embedding(c_n, e_n, activation="tanh")

e1 = encoder(x1)
e2 = encoder(x2)

X = K.layers.merge([e1, e2], 'concat', -1)

X = K.layers.Dense(y_n, activation='sigmoid')

model = K.models.Model(inputs=[x1, x2], outputs=X)


