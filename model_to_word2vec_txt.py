#!/usr/bin/env python

import sys
from gensim.models import Word2Vec

# converts from gensim's word2vec format to google's original format
# used to try out loading the word vectors into postgres

# iterate over the word vector models using something like this:
# find ./word2vec_models -name '*.model' -exec ./model_to_word2vec_txt.py {} \;

def convert_model(model_path):
    print("Processing %s..." % model_path)
    model = Word2Vec.load(model_path)
    replaced_path = model_path.replace(".model", ".w2v.txt")
    model.save_word2vec_format(replaced_path)
    print(" -> Created file %s" % replaced_path)

if __name__ == '__main__':
    convert_model(sys.argv[1])
