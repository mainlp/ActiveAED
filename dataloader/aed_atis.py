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
import random
from typing import List, Tuple, Dict
from pathlib import Path
import json

import datasets

_CITATION = """\
@inproceedings{hemphill-etal-1990-atis,
    title = "The {ATIS} Spoken Language Systems Pilot Corpus",
    author = "Hemphill, Charles T.  and
      Godfrey, John J.  and
      Doddington, George R.",
    booktitle = "Speech and Natural Language: Proceedings of a Workshop Held at Hidden Valley, {P}ennsylvania, June 24-27,1990",
    year = "1990",
    url = "https://aclanthology.org/H90-1021",
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

_DATASETNAME = "aed_atis"

_DESCRIPTION = """\
This dataset is designed for Annotation Error Detection.
"""

_HOMEPAGE = ""

_LICENSE = ""

_URLS = {
    _DATASETNAME: "https://raw.githubusercontent.com/howl-anderson/ATIS_dataset/master/data/standard_format/rasa/train.json",
}

_SOURCE_VERSION = "1.0.0"

_SCHEMA = datasets.Features({
    "id": datasets.Value("string"),
    "text": datasets.Value("string"),
    "label": datasets.Value("string"),
    "true_label": datasets.Value("string"),
})


class AED_ATIS(datasets.GeneratorBasedBuilder):
    _VERSION = datasets.Version(_SOURCE_VERSION)

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
           name="aed_atis_5"
        ),
        datasets.BuilderConfig(
            name="small_aed_atis_5"
        ),
        datasets.BuilderConfig(
            name="aed_atis_10"
        ),
        datasets.BuilderConfig(
            name="small_aed_atis_10"
        ),
    ]
    DEFAULT_CONFIG_NAME = "aed_atis_5"

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

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`

    def _generate_examples(self, data_path: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        with data_path.open() as f:
            data = json.load(f)


        examples = data["rasa_nlu_data"]["common_examples"]

        random.seed(42)
        if self.config.name.startswith("small_"):
            examples = random.sample(examples, 500)

        random.seed(42)
        noise_level = int(self.config.name.split("_")[-1]) / 100
        noise_indices = random.sample(list(range(len(examples))), int(len(examples) * noise_level))
        label_set = sorted(set(i["intent"] for i in examples))
        for i, example in enumerate(examples):
            if i in noise_indices:
                label = random.choice(label_set)
            else:
                label = example["intent"]

            yield (i, {
                "id": str(i),
                "text": example["text"],
                "label": label,
                "true_label": example["intent"]
            })
