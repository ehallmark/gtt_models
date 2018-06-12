import sys
sys.path.insert(0, "/home/ehallmark/repos/gtt_models/src")
from flask import Flask, jsonify
from flask import request
import numpy as np
from src.java_compatibility.BuildPatentEncodings import load_model, load_word2vec_index_maps, norm_across_rows, encode_text, extract_text_model

app = Flask(__name__)

model_file_128 = '/home/ehallmark/data/python/w2v_cpc128_rnn_model_keras128.h5'
encoder = load_model(model_file_128)
model = encoder.model
model.summary()

text_model = extract_text_model(model)
word_idx_map = load_word2vec_index_maps()

@app.route("/encode")
def encode():
    to_encode = request.args.get('text', type=str)
    print('Received text: ', to_encode)
    encoding = []
    if to_encode is not None:
        all_text_encoding = encode_text(text_model, word_idx_map, [to_encode])
        text_norm = norm_across_rows(all_text_encoding)
        all_text_encoding = all_text_encoding / np.where(text_norm != 0, text_norm, 1)[:, np.newaxis]
        if all_text_encoding.shape[0]>0:
            encoding = all_text_encoding.flatten().tolist()
    print('Encoding found: ', encoding)
    return jsonify(encoding)


if __name__ == '__main__':
    app.run(port=5000)


