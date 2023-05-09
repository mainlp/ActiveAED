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
This template serves as a starting point for contributing a dataset to the BigScience Biomedical repo.

When modifying it for your dataset, look for TODO items that offer specific instructions.

Full documentation on writing dataset loading scripts can be found here:
https://huggingface.co/docs/datasets/add_dataset.html

To create a dataset loading script you will create a class and implement 3 methods:
  * `_info`: Establishes the schema for the dataset, and returns a datasets.DatasetInfo object.
  * `_split_generators`: Downloads and extracts data for each split (e.g. train/val/test) or associate local data with each split.
  * `_generate_examples`: Creates examples from data on disk that conform to each schema defined in `_info`.

TODO: Before submitting your script, delete this doc string and replace it with a description of your dataset.

[bigbio_schema_name] = (kb, pairs, qa, text, t2t, entailment)
"""

from typing import List, Tuple, Dict
from pathlib import Path
import json

import datasets

_CITATION = """\
@inproceedings{larson-etal-2020-inconsistencies,
    title = "Inconsistencies in Crowdsourced Slot-Filling Annotations: A Typology and Identification Methods",
    author = "Larson, Stefan  and
      Cheung, Adrian  and
      Mahendran, Anish  and
      Leach, Kevin  and
      Kummerfeld, Jonathan K.",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2020.coling-main.442",
    doi = "10.18653/v1/2020.coling-main.442",
    pages = "5035--5046",
    abstract = "Slot-filling models in task-driven dialog systems rely on carefully annotated training data. However, annotations by crowd workers are often inconsistent or contain errors. Simple solutions like manually checking annotations or having multiple workers label each sample are expensive and waste effort on samples that are correct. If we can identify inconsistencies, we can focus effort where it is needed. Toward this end, we define six inconsistency types in slot-filling annotations. Using three new noisy crowd-annotated datasets, we show that a wide range of inconsistencies occur and can impact system performance if not addressed. We then introduce automatic methods of identifying inconsistencies. Experiments on our new datasets show that these methods effectively reveal inconsistencies in data, though there is further scope for improvement.",
}

"""

_DATASETNAME = "inconsistencies_companies"

_DESCRIPTION = """\
This dataset is designed for Annotation Error Detection.
"""

_HOMEPAGE = ""

_LICENSE = "CC-BY 4.0"

_URLS = {
    _DATASETNAME: "https://drive.google.com/uc?export=download&id=1h6OB8rUmvjQNHl8L6erNjteXvRfI2sZF",
}

_SOURCE_VERSION = "1.0.0"

_SCHEMA = datasets.Features({
    "id": datasets.Value("string"),
    "official_split": datasets.Value("string"),
    "tokens":  datasets.Sequence(datasets.Value("string")),
    "tags_gold": datasets.Sequence(datasets.Value("string")),
    "tags": datasets.Sequence(datasets.Value("string")),
    "classification": datasets.Value("string"),
    "sentence": datasets.Value("string"),
})



class InconsistenciesCompanies(datasets.GeneratorBasedBuilder):

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
        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "data_dir": data_dir,
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`

    def _generate_examples(self, data_dir) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        data_dir = Path(data_dir)
        gold_file = data_dir / "data" / "companies_gold.json"
        crowd_file = data_dir / "data" / "companies_crowd.json"

        with open(gold_file, "r") as f:
            gold = json.load(f)["sentences"]

        with open(crowd_file, "r") as f:
            crowd = json.load(f)["sentences"]


        for idx_example, (gold, crowd) in enumerate(zip(gold, crowd)):
            assert gold["sentence"] == crowd["sentence"]

            yield idx_example, {
                "id": str(idx_example),
                "official_split": gold["official_split"],
                "tokens": [i["word"] for i in gold["svpLabels"]],
                "tags_gold": [i["label"] for i in gold["svpLabels"]],
                "tags": [i["label"] for i in crowd["svpLabels"]],
                "classification": gold["classification"],
                "sentence": gold["sentence"],
            }

if __name__ == '__main__':
    breakpoint()
    data = datasets.load_dataset(__file__)
    import numpy as np
    np.mean([i['tags_gold'] == i["tags_crowd"] for i in data['train']])