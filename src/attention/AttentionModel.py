import numpy as np
from keras.layers import RepeatVector,Concatenate,Dense,Dot,Activation,LSTM,Input,Bidirectional, Lambda, Reshape
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam


class AttentionModelCreator:

    def __init__(self, Fx, Tx, Fy, Ty, n_a, n_s, e1=8, e2=8, activation="tanh"):
        # Defined shared layers as global variables
        self.repeator = RepeatVector(Tx)
        self.Tx = Tx
        self.Fx = Fx
        self.Fy = Fy
        self.Ty = Ty
        self.n_a = n_a
        self.n_s = n_s
        self.concatenator = Concatenate(axis=-1)
        self.densor1 = Dense(e1, activation=activation)
        self.densor2 = Dense(e2, activation=activation)
        # We are using a custom softmax(axis = 1) loaded in this notebook
        self.activator = Activation("softmax", name='attention_weights')
        self.dotor = Dot(axes=1)
        self.a_ = Bidirectional(LSTM(self.n_a, return_sequences=True), merge_mode='concat')
        self.post_activation_LSTM_cell = LSTM(self.n_s, return_state=True)
        self.output_act = 'softmax'
        if self.Fy == 1:
            self.output_act = 'sigmoid'
        self.output_layer = Dense(self.Fy, activation=self.output_act)

    def create_from_inputs(self, loss='categorical_crossentropy',
                           layers_only=False, optimizer=Adam(lr=0.001),
                           x=None, s0=None, c0=None):

        if x is None:
            x = Input(shape=(self.Tx, self.Fx))
        if s0 is None:
            s0 = Input(shape=(self.n_s,), name='s0')
        if c0 is None:
            c0 = Input(shape=(self.n_s,), name='c0')

        def one_step_attention(a, s_prev):
            """
            Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
            "alphas" and the hidden states "a" of the Bi-LSTM.

            Arguments:
            a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
            s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)

            Returns:
            context -- context vector, input of the next (post-attention) LSTM cell
            """

            # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a" (≈ 1 line)
            s_prev = self.repeator(s_prev)
            # Use concatenator to concatenate a and s_prev on the last axis (≈ 1 line)
            concat = self.concatenator([a, s_prev])
            # Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies" variable e. (≈1 lines)
            e = self.densor1(concat)
            # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable energies. (≈1 lines)
            energies = self.densor2(e)
            # Use "activator" on "energies" to compute the attention weights "alphas" (≈ 1 line)
            alphas = self.activator(energies)
            # Use dotor together with "alphas" and "a" to compute the context vector to be given to the next (post-attention) LSTM-cell (≈ 1 line)
            context = self.dotor([alphas, a])

            return context


        def build_model(x):
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

            s = s0
            c = c0

            # Initialize empty list of outputs
            outputs = []

            x = Reshape((self.Tx, self.Fx,))(x)
            #x = K.permute_dimensions(x, (0,2))
            #Lambda(lambda o: K.permute_dimensions(o, (0, 2, 1)))(x)
            a = self.a_(x)

            # Step 2: Iterate for Ty steps
            for t in range(self.Ty):
                #  Step 2.A: Perform one step of the attention mechanism to get back
                #  the context vector at step t (≈ 1 line)
                context = one_step_attention(a, s)

                #  Step 2.B: Apply the post-attention LSTM cell to the "context" vector.
                #  Don't forget to pass: initial_state = [hidden state, cell state] (≈ 1 line)
                s, _, c = self.post_activation_LSTM_cell(context, initial_state=[s, c])

                #  Step 2.C: Apply Dense layer to the hidden state output of the post-attention LSTM (≈ 1 line)
                out = self.output_layer(s)

                #  Step 2.D: Append "out" to the "outputs" list (≈ 1 line)
                outputs.append(out)

            #  Step 3: Create model instance taking three inputs and returning the list of outputs. (≈ 1 line)
            if not layers_only:
                return model
            else:
                return [x, s0, c0], outputs

        # create model
        inputs, outputs = build_model(x)
        if not layers_only:
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
            return model
        else:
            return outputs


def train(model, hidden_units_post_attention, inputs, outputs, epochs=1, batch_size=512):
    m = inputs.shape[0]

    # define inputs
    s0 = np.zeros((m, hidden_units_post_attention))
    c0 = np.zeros((m, hidden_units_post_attention))

    outputs = list(outputs.swapaxes(0,1))
    # train
    model.fit([inputs, s0, c0], outputs, epochs=epochs, batch_size=batch_size)


