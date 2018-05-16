from keras.callbacks import LearningRateScheduler
from src.models.Word2VecModel import Word2Vec
from src.CPC2Vec.Word2Vec import load_cpc_data, build_dictionaries


model_file_32 = '/home/ehallmark/data/python/cpc_sim_model_keras_word2vec_32.h5'
model_file_64 = '/home/ehallmark/data/python/cpc_sim_model_keras_word2vec_64.h5'
model_file_128 = '/home/ehallmark/data/python/cpc_sim_model_keras_word2vec_128.h5'

embedding_size_to_file_map = {
    32: model_file_32,
    64: model_file_64,
    128: model_file_128
}

scheduler = LearningRateScheduler(lambda n: learning_rate/(max(1, n*5)))

(data, val_data) = load_cpc_data(randomize=True)
dictionary, reverse_dictionary = build_dictionaries()
((word_target, word_context), labels) = data
((val_target, val_context), val_labels) = val_data

load_previous_model = True
batch_size = 512
learning_rate = 0

for vector_dim, model_file in embedding_size_to_file_map.items():
    word2vec = Word2Vec(model_file, load_previous_model=load_previous_model,
                        batch_size=batch_size, loss_func='mean_squared_error',
                        embedding_size=vector_dim, lr=learning_rate)
    print("Starting to test model with embedding_size: ", vector_dim)
    print(word2vec.model.test_on_batch([val_target, val_context], val_labels))

