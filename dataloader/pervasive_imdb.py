# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Tuple, Dict
from pathlib import Path
import json
import os
import numpy as np

import datasets

_CITATION = """\
@inproceedings{DBLP:conf/nips/NorthcuttAM21,
  author    = {Curtis G. Northcutt and
               Anish Athalye and
               Jonas Mueller},
  editor    = {Joaquin Vanschoren and
               Sai{-}Kit Yeung},
  title     = {Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks},
  booktitle = {Proceedings of the Neural Information Processing Systems Track on
               Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December
               2021, virtual},
  year      = {2021},
  url       = {https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/f2217062e9a397a1dca429e7d70bc6ca-Abstract-round1.html},
  timestamp = {Thu, 05 May 2022 16:53:59 +0200},
  biburl    = {https://dblp.org/rec/conf/nips/NorthcuttAM21.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

_DATASETNAME = "pervasive_imdb"

_DESCRIPTION = """\
This dataset is designed for Annotation Error Detection.
"""

_HOMEPAGE = ""

_LICENSE = "GPL3"

_URLS = {
    "imdb": "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
    "mturk": "https://raw.githubusercontent.com/cleanlab/label-errors/main/mturk/imdb_mturk.json",
    "indexing": "https://raw.githubusercontent.com/cleanlab/label-errors/main/dataset_indexing/imdb_test_set_index_to_filename.json"
}

_SOURCE_VERSION = "1.0.0"

_SCHEMA = datasets.Features({
    "id": datasets.Value("string"),
    "text": datasets.Value("string"),
    "label": datasets.Value("string"),
    "true_label": datasets.Value("string"),
})


class InconsistenciesFlights(datasets.GeneratorBasedBuilder):
    _VERSION = datasets.Version(_SOURCE_VERSION)

    def _info(self) -> datasets.DatasetInfo:
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=_SCHEMA,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        imdb_dir = dl_manager.download_and_extract(_URLS["imdb"])
        mturk_file = dl_manager.download_and_extract(_URLS["mturk"])
        indexing_file = dl_manager.download_and_extract(_URLS["indexing"])

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "imdb_dir": Path(imdb_dir) / "aclImdb",
                    "mturk_file": Path(mturk_file),
                    "indexing_file": Path(indexing_file)
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`


    def _generate_examples(self, imdb_dir: Path, mturk_file: Path, indexing_file: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        walk_order = {}
        # We don't deal with train set indices, so any order is fine for the train set.
        walk_order['train'] = [d + z for d in ["neg/", "pos/"] \
                               for z in os.listdir(imdb_dir / 'train' / d)]
        # Test set walk order needs to match our order to map errors correctly.
        with open(indexing_file, 'r') as rf:
            walk_order['test'] = json.load(rf)

        # This text dict stores the text data with keys ['train', 'test']
        text = {}
        # Read in text data for IMDB
        for dataset in ['train', 'test']:
            text[dataset] = []
            dataset_dir = imdb_dir / dataset
            for i, fn in enumerate(walk_order[dataset]):
                with open(dataset_dir / fn, 'r') as rf:
                    text[dataset].append(rf.read())

        idx_to_mturk = {}

        with open(mturk_file) as f:
            mturk_data = json.load(f)
            for datapoint in mturk_data:
                idx = walk_order['test'].index(datapoint['id'].removeprefix('test/') + ".txt")
                idx_to_mturk[idx] = datapoint["mturk"]


        # The given labels for both train and test set are the same.
        labels = np.concatenate([np.zeros(12500), np.ones(12500)]).astype(int)

        for i in range(25000):
            if i in idx_to_mturk and idx_to_mturk[i]["given"] < 3:
                true_label = not bool(labels[i])
            else:
                true_label = bool(labels[i])
            yield (i, {
                "id": str(i),
                "text": text["test"][i],
                "label": bool(labels[i]),
                "true_label": true_label
            })
