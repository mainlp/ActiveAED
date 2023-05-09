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

_DATASETNAME = "inconsistencies_flights"

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
    "tokens":  datasets.Sequence(datasets.Value("string")),
    "tags_gold": datasets.Sequence(datasets.Value("string")),
    "tags": datasets.Sequence(datasets.Value("string")),
})



class InconsistenciesFlights(datasets.GeneratorBasedBuilder):

    _VERSION = datasets.Version(_SOURCE_VERSION)

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="inconsistencies_flights"
        ),
        datasets.BuilderConfig(
            name="inconsistencies_flights_random"
        ),
    ]
    DEFAULT_CONFIG_NAME = "inconsistencies_flights"


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

    def _read_examples(self, file_path):
        """
        data looks like this:
        purchase flight leaving { FROM_LOCATION montreal } to { TO_LOCATION yyz }

        return tokens and tags separately
        """
        tokens = []
        tags = []

        with open(file_path, "r") as f:
            cur_tag = "O"
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                line_tokens = []
                line_tags = []
                token_iter = iter(line.split())
                for token in token_iter:
                    if token == "{":
                        tag = next(token_iter).strip()
                        token = next(token_iter).strip()

                        line_tokens.append(token)
                        line_tags.append("B-" + tag)

                        cur_tag = "I-" + tag

                    elif token == "}":
                        cur_tag = "O"
                    else:
                        line_tokens.append(token)
                        line_tags.append(cur_tag)
                    
                tokens.append(line_tokens)
                tags.append(line_tags)
        return tokens, tags

    def _generate_examples(self, data_dir) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        data_dir = Path(data_dir)
        gold_file = data_dir / "data" / "flights_gold.data"
        crowd_file = data_dir / "data" / "flights_crowd.data"


        all_gold_tokens, all_gold_tags = self._read_examples(gold_file)
        all_crowd_tokens, all_crowd_tags = self._read_examples(crowd_file)

        gold_tokens_to_tags = {tuple(tokens): tags for tokens, tags in zip(all_gold_tokens, all_gold_tags)}
        crowd_tokens_to_tags = {tuple(tokens): tags for tokens, tags in zip(all_crowd_tokens, all_crowd_tags)}

        tag_set = set()
        for tags in gold_tokens_to_tags.values():
            tag_set.update(tags)
        tag_set = sorted(tag_set)

        error_prob = 4.1 / 100 # from Klie et al.
        for idx_example, (gold_tokens, gold_tags) in enumerate(gold_tokens_to_tags.items()):

            if self.config.name.endswith("_random"):
                crowd_tags = []
                for tag in gold_tags:
                    if random.uniform(0, 1) < error_prob:
                        crowd_tags.append(random.choice(tag_set))
                    else:
                        crowd_tags.append(tag)
            else:
                crowd_tags = crowd_tokens_to_tags[gold_tokens]

            yield idx_example, {
                "id": str(idx_example),
                "tokens": gold_tokens,
                "tags_gold": gold_tags,
                "tags": crowd_tags,
            }

if __name__ == "__main__":
    data = datasets.load_dataset(__file__, )
    for i in range(10):
        for token, tag in zip(data["train"][i]["tokens"], data["train"][i]["tags_gold"]):
            print(token, tag)
        print()
    