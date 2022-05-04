#!/usr/bin/env python

import pickle
import re

from pathlib import Path
from gensim.models import Word2Vec
from pygtrie import CharTrie

# the root for the word-lapse-models datafiles
data_folder = Path("./")

def get_all_year_models(use_keyedvec=True, make_picked_trie=True):
    model_suffix = ".model"

    def extract_year(k):
        return re.search(r"(\d+)_(\d)[^.]*%s" % model_suffix, str(k)).group(1)

    # first, produce a list of word models sorted by year
    # (groupby requires a sorted list, since it accumulates groups linearly)
    word_models = list(
        (data_folder / Path("word2vec_models")).rglob(f"*/*{model_suffix}")
    )
    word_models_sorted = sorted(word_models, key=extract_year)

    total_vocab = set()

    for model_ref in word_models_sorted:
        print(f"Loading {model_ref}...")
        model = Word2Vec.load(str(model_ref))
        vocab = model.wv.key_to_index.keys()
        unique_vocab = set(vocab)
        print(f" - Unique words: {len(unique_vocab)}")

        total_vocab = total_vocab.union(unique_vocab)

    print(f" => Total unique words: {len(total_vocab)}")

    with open("full_vocab.txt", "w") as fp:
        fp.writelines(f"{x}\n" for x in total_vocab)

    if make_picked_trie:
        vocab_pickle_path = data_folder / Path("vocab_trie.pkl")

        print("Creating vocab trie and pickling it to %s..." % vocab_pickle_path)
        vocab_trie = CharTrie()
        for line in (x.strip() for x in total_vocab if x.strip() != ''):
            vocab_trie[line] = True

        print("Trie size: %d" % len(vocab_trie))
            
        with open(vocab_pickle_path, "wb") as fp:
            pickle.dump(vocab_trie, fp)
        
        print("...done!")

def main():
    get_all_year_models()

if __name__ == '__main__':
    main()
