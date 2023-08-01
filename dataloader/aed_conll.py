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

"""
"""

from typing import List, Tuple, Dict
from pathlib import Path

import datasets

_CITATION = """\
@inproceedings{reiss-etal-2020-identifying,
    title = "Identifying Incorrect Labels in the {C}o{NLL}-2003 Corpus",
    author = "Reiss, Frederick  and
      Xu, Hong  and
      Cutler, Bryan  and
      Muthuraman, Karthik  and
      Eichenberger, Zachary",
    booktitle = "Proceedings of the 24th Conference on Computational Natural Language Learning",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.conll-1.16",
    doi = "10.18653/v1/2020.conll-1.16",
    pages = "215--226",
    abstract = "The CoNLL-2003 corpus for English-language named entity recognition (NER) is one of the most influential corpora for NER model research. A large number of publications, including many landmark works, have used this corpus as a source of ground truth for NER tasks. In this paper, we examine this corpus and identify over 1300 incorrect labels (out of 35089 in the corpus). In particular, the number of incorrect labels in the test fold is comparable to the number of errors that state-of-the-art models make when running inference over this corpus. We describe the process by which we identified these incorrect labels, using novel variants of techniques from semi-supervised learning. We also summarize the types of errors that we found, and we revisit several recent results in NER in light of the corrected data. Finally, we show experimentally that our corrections to the corpus have a positive impact on three state-of-the-art models.",
}
"""

_DATASETNAME = "aed_conll"

_DESCRIPTION = """\
This dataset is designed for Annotation Error Detection.
"""

_HOMEPAGE = ""

_LICENSE = ""

_URLS = {
    _DATASETNAME: "https://drive.google.com/uc?export=download&id=1jiheAs0fa8jCJVr6wGyhPVLnZyes0LCD",
}

_SOURCE_VERSION = "1.0.0"


_SCHEMA = datasets.Features({
    "id": datasets.Value("string"),
    "tokens":  datasets.Sequence(datasets.Value("string")),
    "tags_gold": datasets.Sequence(datasets.Value("string")),
    "tags": datasets.Sequence(datasets.Value("string")),
})


class AED_CONLL(datasets.GeneratorBasedBuilder):
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
        urls = _URLS[_DATASETNAME]
        data_path = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "data_path": Path(data_path),
                },
            ),
        ]


    def read_conll(self, path):
        tokens_to_tags = {}

        tokens = []
        tags = []

        with path.open() as f:
            for line in f:
                line = line.strip()
                if line.startswith("-DOCSTART-"):
                    continue

                if not line:
                    tokens_to_tags[tuple(tokens)] = tuple(tags)
                    tokens = []
                    tags = []
                    continue

                fields = line.split()
                tokens.append(fields[0])
                tags.append(fields[-1])

            if tokens:
                tokens_to_tags[tuple(tokens)] = tuple(tags)

        return tokens_to_tags


    def _generate_examples(self, data_path: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        data_path = data_path / "aed_conll"
        original = {}
        corrected = {}

        original.update(self.read_conll(data_path / "original_corpus" / "eng.train"))
        original.update(self.read_conll(data_path / "original_corpus" / "eng.testa"))
        original.update(self.read_conll(data_path / "original_corpus" / "eng.testb"))

        corrected.update(self.read_conll(data_path / "corrected_corpus" / "eng.train"))
        corrected.update(self.read_conll(data_path / "corrected_corpus" / "eng.testa"))
        corrected.update(self.read_conll(data_path / "corrected_corpus" / "eng.testb"))

        for i, tokens in enumerate(original):
            if not tokens or tokens not in corrected:
                continue
            yield (i, {
                "id": str(i),
                "tokens": tokens,
                "tags": original[tokens],
                "tags_gold": corrected[tokens]
            })
