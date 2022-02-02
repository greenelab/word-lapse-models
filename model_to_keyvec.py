#!/usr/bin/env python

import sys
from gensim.models import Word2Vec, KeyedVectors

# advice taken from https://stackoverflow.com/a/43067907

# iterate over the word vector models using something like this:
# find ./word2vec_models -name '*.model' -exec ./model_to_keyvec.py {} \;

def convert_model(model_path):
    print("Processing %s..." % model_path)
    model = Word2Vec.load(model_path)
    replaced_path = model_path.replace(".model", ".wordvectors")
    model.wv.save(replaced_path)
    print(" -> Created file %s" % replaced_path)

if __name__ == '__main__':
    convert_model(sys.argv[1])
