#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Convert word embedding to hdf5 file
conv_wemb2hdf5.py word.dict word_embedding.txt word_embedding.hdf5
"""
import h5py
import numpy as np

dictfile = "lcsts-char.comb.dict"
word_embeddings_text_file = "w2v-lcsts-all.txt"
outfile = "word_embedding.hdf5"

def get_w2v(vecfilename, wordlist):
    np.random.seed(1337)
    with open(vecfilename) as vec_file:
        lines = vec_file.readlines()[1:]
    lines = [l.strip().split() for l in lines]
    words_to_vec = dict()
    for l in lines:
        word, vecs = l[0], l[1:]
        words_to_vec[word] = [float(v) for v in vecs]
    word_vec_dim = len(words_to_vec[words_to_vec.keys()[0]])
    ret_array = 0.01 * np.random.random_sample((len(wordlist), word_vec_dim)) - 0.01
    ret_array[0] = np.zeros(word_vec_dim)
    ret_array[1] = np.zeros(word_vec_dim)
    ret_array[2] = np.zeros(word_vec_dim)
    ret_array[3] = np.zeros(word_vec_dim)
    oov_count = 0
    for w in wordlist:
        if w in words_to_vec:
            ret_array[wordlist[w] - 1] = words_to_vec[w]
        else:
            oov_count += 1
            # print("Word %s not found, randomized." % w)
    # print("OOVs: %s" % oov_count)
    return ret_array

word_dict = dict()
with open(dictfile) as dFile:
    words_and_ids = [l.strip().split() for l in dFile.readlines()]
for w_n_id in words_and_ids:
    word_dict[w_n_id[0]] = int(w_n_id[1])
word_embeddings = get_w2v(word_embeddings_text_file, word_dict)
f = h5py.File(outfile, "w")
f['word_vecs'] = np.array(word_embeddings, dtype=np.float32)
f.close()
