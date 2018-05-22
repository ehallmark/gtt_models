import numpy as np


def convert_to_h5(filename, outfile, delimiter=' '):
    print('loading: ', filename)
    data = np.loadtxt(filename, delimiter=delimiter)
    if data is not None:
        print('saving to ', outfile)
        np.save(outfile, data)
    else:
        print('could not find file.')


if __name__ == "__main__":
    file = '/home/ehallmark/Downloads/word2vec256_vectors.txt'
    outfile = '/home/ehallmark/Downloads/word2vec256_vectors.h5'
    convert_to_h5(file, outfile)