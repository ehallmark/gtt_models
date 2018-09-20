import numpy as np
import pandas as pd
from keras.layers import Embedding, Reshape, LSTM, Input, Dense, Concatenate, Bidirectional, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from sklearn import metrics
import re


def test_model(model, x, y):
    y_pred = model.predict(x)
    return metrics.log_loss(y, y_pred)


def convert_sentences_to_inputs(sentences, word_to_index_map, max_sequence_length):
    x = np.zeros((len(sentences), max_sequence_length))
    for i in range(len(sentences)):
        sentence = sentences.iloc[i]
        words = re.sub(r'\W+', ' ', str(sentence).lower()).split("\\s+")
        idx = 0
        for j in range(max_sequence_length-len(words), max_sequence_length, 1):
            word = words[idx]
            idx += 1
            if word in word_to_index_map:
                x[i, j] = word_to_index_map[word]+1  # don't forget to add 1 to account for mask at index 0
    return x.reshape((len(sentences), max_sequence_length, 1))


def binary_to_categorical(bin):
    x = np.zeros((len(bin), 2))
    for i in range(len(bin)):
        x[i, int(bin[i])] = 1.0
    return x


def label_for_row(row):
    if row['controversiality'] > 0 or row['score'] < 0:
        return 1
    else:
        return 0


def get_data(max_sequence_length, word_to_index_map, num_validations=25000):
    # load the data used to train/validate model
    print('Loading data...')
    x0 = pd.read_csv('/home/ehallmark/Downloads/comment_comments0.csv', sep=',')
    x1 = pd.read_csv('/home/ehallmark/Downloads/comment_comments1.csv', sep=',')

    x = x0.append(x1, ignore_index=True)
    x = x.sample(frac=1, replace=False)

    x_val = x[-num_validations:]
    x = x[0:-num_validations]

    print('Train shape:', x.shape)
    print('Val shape: ', x_val.shape)

    y = binary_to_categorical(np.array([label_for_row(row) for _, row in x.iterrows()]).astype(np.int32))
    y_val = binary_to_categorical(np.array([label_for_row(row) for _, row in x_val.iterrows()]).astype(np.int32))

    print('Converting sentences for training data...')
    x1 = convert_sentences_to_inputs(x['parent_text'], word_to_index_map, max_sequence_length)
    x2 = convert_sentences_to_inputs(x['text'], word_to_index_map, max_sequence_length)

    print('Converting sentences for validation data...')
    x1_val = convert_sentences_to_inputs(x_val['parent_text'], word_to_index_map, max_sequence_length)
    x2_val = convert_sentences_to_inputs(x_val['text'], word_to_index_map, max_sequence_length)

    print('Finished loading data...')
    return ([x1, x2], y), ([x1_val, x2_val], y_val)


def load_word2vec_index_maps(word2vec_index_file):
    word2vec_index = np.array(pd.read_csv(word2vec_index_file, delimiter=',', header=None))
    word_to_index_map = {}
    index_to_word_map = {}
    for row in word2vec_index:
        idx = int(row[1])
        word = str(row[0])
        if idx == 0:
            print("Found 0 index in word2vec... 0 index should be the mask index (i.e. not a word)."
                  + " Try rerunning IngestWord2VecToText java program.")
            raise ArithmeticError
        word_to_index_map[word] = idx
        index_to_word_map[idx] = word
    return word_to_index_map, index_to_word_map


def load_word2vec_model_layer(model_file, sequence_length):
    # load word2vec into keras embedding layer
    print('Loading word2vec model...')
    vocab_matrix = np.load(model_file)

    vocab_len = vocab_matrix.shape[0]
    dim = vocab_matrix.shape[1]

    # init layer - remember that mask_zero = False, but the zero index is a zero vector so all indices for actual
    #   words must be incremented by 1
    embedding_layer = Embedding(vocab_len, dim, input_length=sequence_length, mask_zero=False, trainable=False)

    # build layer
    embedding_layer.build((None,))

    # set weights
    embedding_layer.set_weights([vocab_matrix])

    return embedding_layer


if __name__ == "__main__":
    vocab_vector_file_h5 = '/home/ehallmark/data/python/word2vec256_vectors.h5.npy'  # h5 extension faster? YES by alot
    word2vec_index_file = '/home/ehallmark/data/python/word2vec256_index.txt'
    model_file = '/home/ehallmark/data/python/controversy_model.nn'

    initial_epoch = 0  # allows resuming training from particular epoch
    word2vec_size = 256  # size of the embedding
    max_sequence_length = 96  # max number of words to consider in the comment
    learning_rate = 0.00001  # defines the learning rate (initial) for the model
    decay = 0.01  # weight decay
    batch_size = 256  # defines the mini batch size
    epochs = 10  # defines the number of full passes through the training data
    hidden_layer_size = 512  # defines the hidden layer size for the model's layers
    ff_hidden_layer_size = 2048
    num_validations = 50000  # defines the number of training cases to set aside for validation

    # the embedding layer
    word_to_index_map, index_to_word_map = load_word2vec_index_maps(word2vec_index_file)
    embedding_layer = load_word2vec_model_layer(model_file=vocab_vector_file_h5, sequence_length=max_sequence_length)

    # build model
    x1_orig = Input(shape=(max_sequence_length, 1), dtype=np.int32)
    x2_orig = Input(shape=(max_sequence_length, 1), dtype=np.int32)

    x1 = embedding_layer(x1_orig)
    x2 = embedding_layer(x2_orig)
    x1 = Reshape((max_sequence_length, word2vec_size))(x1)
    x2 = Reshape((max_sequence_length, word2vec_size))(x2)

    x1 = LSTM(hidden_layer_size, activation='tanh', return_sequences=False)(x1)
    x2 = LSTM(hidden_layer_size, activation='tanh', return_sequences=False)(x2)

    model = Dense(ff_hidden_layer_size, activation='tanh')(Concatenate()([x1, x2]))
    model = BatchNormalization()(model)
    model = Dense(ff_hidden_layer_size, activation='tanh')(model)
    model = BatchNormalization()(model)
    model = Dense(ff_hidden_layer_size, activation='tanh')(model)
    model = BatchNormalization()(model)
    model = Dense(2, activation='softmax')(model)

    # compile model
    model = Model(inputs=[x1_orig, x2_orig], outputs=model)
    model.compile(loss="binary_crossentropy", optimizer=Adam(lr=learning_rate, decay=decay), metrics=['accuracy'])
    model.summary()

    # load training data
    print('Getting data...')
    (data, data_val) = get_data(max_sequence_length, word_to_index_map, num_validations)

    x, y = data

    # train model
    avg_error = test_model(model, data_val[0], data_val[1])
    print("Starting model score: ", avg_error)
    prev_error = avg_error
    best_error = avg_error
    errors = list()
    errors.append(avg_error)
    for i in range(epochs):
        model.fit(x, y, batch_size=batch_size, initial_epoch=initial_epoch + i, epochs=initial_epoch + i + 1,
                  validation_data=data_val, shuffle=True)
        avg_error = test_model(model, data_val[0], data_val[1])

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


