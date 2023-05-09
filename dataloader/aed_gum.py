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

import random
from typing import List, Tuple, Dict
from pathlib import Path
import pyconll

import datasets

_CITATION = """\
@article{zeldes2017gum,
  title={The GUM corpus: Creating multilayer resources in the classroom},
  author={Zeldes, Amir},
  journal={Language Resources and Evaluation},
  volume={51},
  number={3},
  pages={581--612},
  year={2017},
  publisher={Springer}
}

@article{10.1162/coli_a_00464,
    author = {Klie, Jan-Christoph and Webber, Bonnie and Gurevych, Iryna},
    title = "{Annotation Error Detection: Analyzing the Past and Present for a More Coherent Future}",
    journal = {Computational Linguistics},
    pages = {1-42},
    year = {2022},
    month = {11},
    abstract = "{Annotated data is an essential ingredient in natural language processing for training and evaluating machine learning models. It is therefore very desirable for the annotations to be of high quality. Recent work, however, has shown that several popular datasets contain a surprising number of annotation errors or inconsistencies. To alleviate this issue, many methods for annotation error detection have been devised over the years. While researchers show that their approaches work well on their newly introduced datasets, they rarely compare their methods to previous work or on the same datasets. This raises strong concerns on methods’ general performance and makes it difficult to asses their strengths and weaknesses. We therefore reimplement 18 methods for detecting potential annotation errors and evaluate them on 9 English datasets for text classification as well as token and span labeling. In addition, we define a uniform evaluation setup including a new formalization of the annotation error detection task, evaluation protocol and general best practices. To facilitate future research and reproducibility, we release our datasets and implementations in an easy-to-use and open source software package.}",
    issn = {0891-2017},
    doi = {10.1162/coli_a_00464},
    url = {https://doi.org/10.1162/coli\_a\_00464},
    eprint = {https://direct.mit.edu/coli/article-pdf/doi/10.1162/coli\_a\_00464/2057485/coli\_a\_00464.pdf},
}
"""

_DATASETNAME = "aed_gum"

_DESCRIPTION = """\
This dataset is designed for Annotation Error Detection.
"""

_HOMEPAGE = ""

_LICENSE = ""

_URLS = {
    "train": "https://github.com/UniversalDependencies/UD_English-GUM/raw/master/en_gum-ud-dev.conllu",
    "dev": "https://github.com/UniversalDependencies/UD_English-GUM/raw/master/en_gum-ud-train.conllu",
    "test": "https://github.com/UniversalDependencies/UD_English-GUM/raw/master/en_gum-ud-test.conllu",
}

_SOURCE_VERSION = "1.0.0"


_SCHEMA = datasets.Features({
    "id": datasets.Value("string"),
    "tokens":  datasets.Sequence(datasets.Value("string")),
    "tags_gold": datasets.Sequence(datasets.Value("string")),
    "tags": datasets.Sequence(datasets.Value("string")),
})

_UPOS_TAG_SET = [
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CCONJ",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "SYM",
    "VERB",
    "X",
]


class AED_GUM(datasets.GeneratorBasedBuilder):
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
        data_paths = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "data_path": Path(data_paths["train"]),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "data_path": Path(data_paths["dev"]),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "data_path": Path(data_paths["test"]),
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`


    def _generate_examples(self, data_path: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        random.seed(42)
        noise_level = 5 / 100

        data = pyconll.load_from_file(str(data_path))
        for i, sentence in enumerate(data):
            tokens = []
            tags = []
            tags_gold = []

            for token in sentence:
                tag_gold = token.upos if token.upos else "NONE"
                if random.uniform(0, 1) < noise_level:
                    tags.append(random.choice(_UPOS_TAG_SET))
                else:
                    tags.append(tag_gold)

                tags_gold.append(tag_gold)
                tokens.append(token.form)


            yield (i, {
                "id": sentence.id,
                "tokens": tokens,
                "tags": tags,
                "tags_gold": tags_gold
            })
