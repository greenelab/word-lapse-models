#!/usr/bin/env python

import re

from pathlib import Path
from gensim.models import Word2Vec

# the root for the word-lapse-models datafiles
data_folder = Path("./")

def get_all_year_models(use_keyedvec=True):
    model_suffix = "model"

    def extract_year(k):
        return re.search(r"(\d+)_(\d)[^.]*\.%s" % model_suffix, str(k)).group(1)

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

def main():
    get_all_year_models()

if __name__ == '__main__':
    main()
