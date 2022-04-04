#!/usr/bin/env python

import pickle
import lzma
from csv import DictReader
from pathlib import Path

from tqdm import tqdm
from pygtrie import CharTrie

# the root for the word-lapse-models datafiles
data_folder = Path("./")

# the number of concept map items, determined by unzipping all_concept_ids.tsv.xz
# and counting the number of lines (-1 for the header)
# (this is used to display a progress bar with an estimate of the processing time)
APPROX_CONCEPT_LINES = 39446070

def get_concept_lines():
    with lzma.open(data_folder / Path("all_concept_ids.tsv.xz"), "rt") as fp:
        reader = DictReader(
            fp, dialect="excel-tab", fieldnames=("concept_id", "concept")
        )
        for row in reader:
            yield row

def create_concept_trie():
    concept_trie_pickle = data_folder / Path("concept_trie.pkl")

    print("regenerating concept trie (this will take a while)...")
    concept_trie = CharTrie()
    for row in tqdm(get_concept_lines(), total=APPROX_CONCEPT_LINES):
        concept_trie[row["concept"]] = row["concept_id"]

    print("...writing pickle...")
    with open(concept_trie_pickle, "wb") as fp:
        pickle.dump(concept_trie, fp)
    print("...done!")

if __name__ == '__main__':
    create_concept_trie()
