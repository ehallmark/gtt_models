from keras.models import Model
import keras as k
from keras.optimizers import Adam
import numpy as np
from src.controversy.Model import convert_sentences_to_ff, convert_sentences_to_rnn, max_sequence_length, load_word2vec_index_maps, word2vec_index_file
import pandas as pd


model_file = '/home/ehallmark/data/python/controversy_model2.nn'


def predict_probability_controversial(parent_text, text, parent_comment_score, model, word_to_index_map,
                                      max_sequence_length, dictionary):
    x1 = convert_sentences_to_rnn(np.array([[parent_text]]), word_to_index_map, max_sequence_length)
    x2 = convert_sentences_to_rnn(np.array([[text]]), word_to_index_map, max_sequence_length)
    x3 = convert_sentences_to_ff(np.array([[parent_text]]), dictionary)
    x4 = convert_sentences_to_ff(np.array([[text]]), dictionary)
    x5 = np.array([[parent_comment_score]])

    features = [
        x1,
        x2,
        x3,
        x4,
        #x5
    ]

    y_hat = model.predict(features)
    return y_hat[0][1]


if __name__ == '__main__':
    # doesn't matter
    learning_rate = 0.00001  # defines the learning rate (initial) for the model
    decay = 0.01  # weight decay

    # load model
    model = k.models.load_model(model_file, compile=False)
    model.compile(loss="binary_crossentropy", optimizer=Adam(lr=learning_rate, decay=decay), metrics=['accuracy'])

    word_to_index_map, index_to_word_map = load_word2vec_index_maps(word2vec_index_file)
    words = pd.read_csv('/home/ehallmark/Downloads/words.csv', sep=',')['word'].values
    dictionary = {}
    for i in range(len(words)):
        dictionary[words[i]] = i

    # samples
    samples = [
        [" Are you retarded?  You missed the point.  He's saying that it isn't just about women's rights, but also about the rights of the father.  Yes, its her right, but where does the other half the child's chromosomes fit into the picture? ", "They fit rather neatly into the mother's uterus. If they were somewhere else, abortion wouldn't be her business either.", 5],
        ["Do you remember the feeling of frustration when your girlfriend pisses you off, disappoints you, or just plain annoys you? Well, no more of that. It's all about you when you're single. Time for yourself, reconnect with friends, and the bachelor life.", "LAN PARTY TIME!", 74],
        ["I'm building a gaming computer -- she's never been into gaming, so I'm taking this opportunity to spend money on stuff I never could have justified before.", "Just steer clear of the opiate called \"World of Warcraft\" if you are easily addicted to stuff...", 65],
        ["A true minecraft fan!  | &gt; he didn't have long to enjoy his ill-gotten gains: **leaving a trail of evidence**, the man was arrested by Detective Pat Hankins and charged with felony burglary", "Guy probably used cobblestone in place of rock. Pssh.", 130],
        ["Haha I dont think Im awesome, but ask any of your girl friends who dye their hair how hard it is to go lighter/blonde. I had to make this white under it, with no darker/orange spots without my hair falling out. This was a once in a lifetime opportunity haha. If this were a guy on here it wouldnt be such a problem :| Im just glad I got to do it all by myself and it worked :)", "you put stuff on your hair, waited, washed it out, put more stuff on, waited, and washed it out. great fucking job, here's some more attention for you. go ahead and get that last word in, too. you know you need it so you can feel like you've won.", 0]
    ]

    for sample in samples:
        controversiality_prob = predict_probability_controversial(sample[0], sample[1], sample[2], model,
                                                                  word_to_index_map, max_sequence_length, dictionary)

        print('---------------------------------------------------------------------------------')
        print('Controversiality: ', controversiality_prob)
        print('\tOriginal Comment:', sample[0], '\n\tResponse:', sample[1], '\n\n')
