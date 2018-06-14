import sys
sys.path.insert(0, "/home/ehallmark/repos/gtt_models")
from flask import Flask, jsonify
from flask import request
import numpy as np
import tensorflow as tf
from src.java_compatibility.BuildPatentEncodings import load_model, load_word2vec_index_maps, norm_across_rows, encode_text, extract_text_model

app = Flask(__name__)

model_file_128 = '/home/ehallmark/data/python/w2v_cpc128_rnn_model_keras128.h5'
encoder = load_model(model_file_128)
model = encoder.model
model.summary()

text_model = extract_text_model(model)
graph = tf.get_default_graph()
word_idx_map, index_to_word_map = load_word2vec_index_maps()
print('Num words: ', len(word_idx_map))


@app.route('/encode', methods=['GET', 'POST'])
def encode():
    to_encode = request.args.get('text', type=str)
    print('Received text: ', to_encode)
    encoding = []
    if to_encode is not None:
        with graph.as_default():
            all_text_encoding = encode_text(text_model, word_idx_map, [to_encode])
        print('Enc: ', all_text_encoding)
        text_norm = norm_across_rows(all_text_encoding)
        all_text_encoding = all_text_encoding / np.where(text_norm != 0, text_norm, 1)[:, np.newaxis]
        if all_text_encoding.max() == all_text_encoding.min():
            encoding = []
        elif all_text_encoding.shape[0] > 0:
            encoding = all_text_encoding.flatten().tolist()
    print('Encoding found: ', encoding)
    return jsonify(encoding)


if __name__ == '__main__':
    app.run(port=5000)


