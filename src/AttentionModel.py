import keras as K
import numpy as np
from keras.layers import RepeatVector,Concatenate,Dense,Dot,Activation,LSTM,Input,Bidirectional
from keras.models import Model


def attention_model(Tx, Fx, Ty, Fy, n_a=64, n_s=32, e1=10, e2=1, activation="tanh",
                    loss='categorical_crossentropy', optimizer=K.optimizers.Adam()):

    # Defined shared layers as global variables
    repeator = RepeatVector(Tx)
    concatenator = Concatenate(axis=-1)
    densor1 = Dense(e1, activation=activation)
    densor2 = Dense(e2, activation=activation)
    activator = Activation("softmax", axis=-1, name='attention_weights') # We are using a custom softmax(axis = 1) loaded in this notebook
    dotor = Dot(axes=1)

    def one_step_attention(a, s_prev):
        """
        Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
        "alphas" and the hidden states "a" of the Bi-LSTM.

        Arguments:
        a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
        s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)

        Returns:
        context -- context vector, input of the next (post-attetion) LSTM cell
        """

        # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a" (≈ 1 line)
        s_prev = repeator(s_prev)
        # Use concatenator to concatenate a and s_prev on the last axis (≈ 1 line)
        concat = concatenator([a, s_prev])
        # Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies" variable e. (≈1 lines)
        e = densor1(concat)
        # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable energies. (≈1 lines)
        energies = densor2(e)
        # Use "activator" on "energies" to compute the attention weights "alphas" (≈ 1 line)
        alphas = activator(energies)
        # Use dotor together with "alphas" and "a" to compute the context vector to be given to the next (post-attention) LSTM-cell (≈ 1 line)
        context = dotor([alphas, a])

        return context

    post_activation_LSTM_cell = LSTM(n_s, return_state=True)
    output_layer = Dense(Fy, activation="softmax")

    def model(Tx, Ty, n_a, n_s):
        """
        Arguments:
        Tx -- length of the input sequence
        Ty -- length of the output sequence
        n_a -- hidden state size of the Bi-LSTM
        n_s -- hidden state size of the post-attention LSTM
        human_vocab_size -- size of the python dictionary "human_vocab"
        machine_vocab_size -- size of the python dictionary "machine_vocab"

        Returns:
        model -- Keras model instance
        """

        # Define the inputs of your model with a shape (Tx,)
        # Define s0 and c0, initial hidden state for the decoder LSTM of shape (n_s,)
        X = Input(shape=(Tx,Fx))
        s0 = Input(shape=(n_s,), name='s0')
        c0 = Input(shape=(n_s,), name='c0')
        s = s0
        c = c0

        # Initialize empty list of outputs
        outputs = []

        # Step 1: Define your pre-attention Bi-LSTM. Remember to use return_sequences=True. (≈ 1 line)
        a = Bidirectional(LSTM(n_a, return_sequences=True), merge_mode='concat')(X)

        # Step 2: Iterate for Ty steps
        for t in range(Ty):
            # Step 2.A: Perform one step of the attention mechanism to get back the context vector at step t (≈ 1 line)
            context = one_step_attention(a, s)

            # Step 2.B: Apply the post-attention LSTM cell to the "context" vector.
            # Don't forget to pass: initial_state = [hidden state, cell state] (≈ 1 line)
            s, _, c = post_activation_LSTM_cell(context, initial_state=[s, c])

            # Step 2.C: Apply Dense layer to the hidden state output of the post-attention LSTM (≈ 1 line)
            out = output_layer(s)

            # Step 2.D: Append "out" to the "outputs" list (≈ 1 line)
            outputs.append(out)

        # Step 3: Create model instance taking three inputs and returning the list of outputs. (≈ 1 line)
        model = Model(inputs=[X, s0, c0], outputs=outputs)
        return model

    # create model
    model = model(Tx, Ty, n_a, n_s)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model


def train(model, hidden_units_post_attention, inputs, outputs, epochs=1):
    m = inputs.shape[0]

    # define inputs
    s0 = np.zeros((m, hidden_units_post_attention))
    c0 = np.zeros((m, hidden_units_post_attention))

    outputs = list(outputs.swapaxes(0,1))
    # train
    model.fit([inputs, s0, c0], outputs, epochs=epochs, batch_size=100)


# example usage
input_sequence_length = 4
input_feature_length = 3
output_sequence_length = 4
output_feature_length = 3
hidden_units_pre_attention = 32
hidden_units_post_attention = 64
attention_model = attention_model(input_sequence_length, input_feature_length,
                                  output_sequence_length, output_feature_length,
                                  n_a=hidden_units_pre_attention, n_s=hidden_units_post_attention,
                                  e1=20,e2=20,
                                  optimizer=K.optimizers.adam(0.001),
                                  loss="mean_squared_error")

X = np.random.uniform(0, 1, (100000, input_sequence_length, input_feature_length)) > 0.5
Y = X.copy()
print(Y[0:10])
print(attention_model.summary())

# train
train(attention_model, hidden_units_post_attention, inputs=X, outputs=Y, epochs=1000)

