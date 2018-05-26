from flask import Flask
from flask import request
from src.models.Word2VecModel import load_cpc_to_index_map
import psycopg2
import pandas as pd

app = Flask(__name__)

@app.route("/encode")
def encode():
    to_encode = request.args.getlist('data[]', type=str)
    print(to_encode)
    return ', '.join(to_encode)


if __name__ == '__main__':
    app.run(port=5000)


