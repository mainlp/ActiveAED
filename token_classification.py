import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Union, Optional

import datasets
import pandas as pd
import seqeval
from pytorch_lightning.utilities.seed import seed_everything
from scipy.stats import rankdata
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, average_precision_score, f1_score, precision_recall_curve
from sklearn.model_selection import KFold, train_test_split
import transformers
import argparse
import torch
from tqdm import tqdm, trange
from transformers import PreTrainedTokenizerBase
import numpy as np
import evaluate
from transformers.data.data_collator import DataCollatorMixin
from transformers.utils import PaddingStrategy

base_dir = Path(__file__).parent



@dataclass
class DataCollatorForTokenClassification(DataCollatorMixin):
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def torch_call(self, features):

        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if labels is None:
            return batch

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch[label_name] = [
                list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
            ]
        else:
            batch[label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + list(label) for label in labels
            ]

        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        batch["position_ids"] = torch.stack([torch.arange(sequence_length) for _ in batch["input_ids"]])

        return batch

class LogitsGetterCallback(transformers.TrainerCallback):
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.train_outputs = []
        self.test_outputs = []
        self.trainer = None

    def on_epoch_end(self, args, state, control, **kwargs):
        self.train_outputs.append(self.trainer.predict(self.train_data))
        self.test_outputs.append(self.trainer.predict(self.test_data))


def score_aum(outputs):
    scores = np.zeros(len(outputs[0].predictions))
    for output in outputs:
        for i_example,  (logits, labels) in enumerate(zip(output.predictions, output.label_ids)):
            example_scores = []
            # TODO: vectorize the inner loop over labels
            for i, label in enumerate(labels):
                if label == -100:
                    continue
                probs = torch.softmax(torch.tensor(logits[i].astype(float)), dim=0)
                prob_true = probs[labels[i]].item()
                probs[labels[i]] = 0
                prob_max_other = max(probs)
                example_scores.append(prob_max_other - prob_true)
            scores[i_example] += np.max(example_scores).item()

    return scores

def score_aum_logits(outputs):
    scores = np.zeros(len(outputs[0].predictions))
    for output in outputs:
        for i_example,  (logits, labels) in enumerate(zip(output.predictions, output.label_ids)):
            example_scores = []
            # TODO: vectorize the inner loop over labels
            for i, label in enumerate(labels):
                if label == -100:
                    continue
                logits_i = torch.tensor(logits[i].astype(float))
                prob_true = logits_i[labels[i]].item()
                logits_i[labels[i]] = -100000
                prob_max_other = max(logits_i)
                example_scores.append(prob_max_other - prob_true)
            scores[i_example] += np.max(example_scores).item()

    return scores


def score_dm(outputs):
    scores = np.zeros(len(outputs[0].predictions))
    for output in outputs:
        for i_example,  (logits, labels) in enumerate(zip(output.predictions, output.label_ids)):
            example_scores = []
            for i, label in enumerate(labels):
                if label == -100:
                    continue
                probs = torch.softmax(torch.tensor(logits[i]).float(), dim=0)
                prob_true = probs[label].item()
                example_scores.append(1 - prob_true) # higher -> more likely to be error
            scores[i_example] += np.max(example_scores).item()

    return scores

def score_cu(outputs):
    output = outputs[-1]
    scores = np.zeros(len(outputs[0].predictions))
    for i_example,  (logits, labels) in enumerate(zip(output.predictions, output.label_ids)):
        example_scores = []
        for i, label in enumerate(labels):
            if label == -100:
                continue
            probs = torch.softmax(torch.tensor(logits[i]).float(), dim=0)
            prob_true = probs[label].item()
            example_scores.append(1 - prob_true) # higher -> more likely to be error
        scores[i_example] = np.max(example_scores).item()

    return scores

def score_retag(outputs):
    scores = np.zeros(len(outputs[0].predictions))
    output = outputs[-1]
    for i_example,  (logits, labels) in enumerate(zip(output.predictions, output.label_ids)):
        example_scores = []
        for i, label in enumerate(labels):
            if label == -100:
                continue
            probs = torch.softmax(torch.tensor(logits[i]), dim=0)
            prob_true = probs[labels[i]].item()
            probs[labels[i]] = 0
            prob_max_other = max(probs)
            example_scores.append(prob_max_other - prob_true)
        scores[i_example] = np.max(example_scores).item()

    return scores



def predict(dataset, label2id, id2label, model_name, out_dir, num_epochs, random_seed=42):

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    cv = KFold(10, shuffle=True, random_state=random_seed)

    error_scores = {
        "aum_train": np.zeros(len(dataset)),
        "aum_test": np.zeros(len(dataset)),
        "dm_train": np.zeros(len(dataset)),
        "dm_test": np.zeros(len(dataset)),
        "cu_train": np.zeros(len(dataset)),
        "cu_test": np.zeros(len(dataset)),
    }
    preds = None


    i = 0
    for train_index, test_index in tqdm(list(cv.split(dataset["tokens"])), desc="CV"):
        model = transformers.AutoModelForTokenClassification.from_pretrained(model_name,
                                                                             num_labels=len(label2id),
                                                                             label2id=label2id,
                                                                             id2label=id2label)

        train_dataset = dataset.select(train_index)
        test_dataset = dataset.select(test_index)

        train_dataset = train_dataset.map(partial(tokenize_and_align_labels, tag_type="tags"), batched=True, )
        test_dataset = test_dataset.map(partial(tokenize_and_align_labels, tag_type="tags"), batched=True, )

        if torch.cuda.is_available():
            batch_size_denominator = max(1, torch.cuda.device_count()) # Make batch size consistent across computation environments with different numbers of GPUs
        else:
            batch_size_denominator = 1
        batch_size = 64

        logits_getter = LogitsGetterCallback(train_data=train_dataset, test_data=test_dataset)

        training_args = transformers.TrainingArguments(
            output_dir=out_dir,
            learning_rate=5e-5,
            per_device_train_batch_size= batch_size // batch_size_denominator,
            num_train_epochs=num_epochs,
            fp16=True if torch.cuda.is_available() else False,
            report_to=[],
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            use_mps_device=False,
            save_total_limit=1,
            # eval_steps=10,

        )

        trainer = transformers.Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            callbacks=[logits_getter]
        )
        logits_getter.trainer = trainer

        trainer.train()

        outputs = trainer.predict(test_dataset)

        error_scores["aum_logits_train"][train_index] += score_aum_logits(logits_getter.train_outputs)
        error_scores["aum_train"][train_index] += score_aum(logits_getter.train_outputs)
        error_scores["dm_train"][train_index] += score_dm(logits_getter.train_outputs)
        error_scores["cu_train"][train_index] += score_cu(logits_getter.train_outputs)

        error_scores["aum_logits_test"][test_index] = score_aum_logits(logits_getter.test_outputs + [outputs])
        error_scores["aum_test"][test_index] = score_aum(logits_getter.test_outputs + [outputs])
        error_scores["dm_test"][test_index] = score_dm(logits_getter.test_outputs + [outputs])
        error_scores["cu_test"][test_index] = score_cu(logits_getter.test_outputs + [outputs])

        # if preds is None:
        #     preds = np.zeros((len(dataset, ) + outputs.predictions.shape[1:]))
        #     preds[:] = -100
        # preds[test_index] = outputs.predictions

    return error_scores, preds


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def tokenize_and_align_labels(examples, tag_type):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(tqdm(examples[tag_type])):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                l = label[word_idx]
                label_ids.append(label2id[l])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels

    return tokenized_inputs


def update_labels(example, ids):
    if example["id"] in ids:
        example["tags"] = example["tags_gold"]
    else:
        example["tags"] = example["tags"]

    return example




seqeval = evaluate.load("seqeval")

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--out_dir", required=True, type=Path)
parser.add_argument("--scorer", required=True)
parser.add_argument("--n_epochs", required=True, type=int)
parser.add_argument("--k", default=50, type=int)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--model_name", default="distilroberta-base")
parser.add_argument("--aggregation", choices=["train", "test", "both"], default="both")
parser.add_argument("--no_active", action="store_true")
parser.add_argument("--dataset_config")
args = parser.parse_args()

seed_everything(args.seed)

tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name, add_prefix_space=True)
dataloader_path = str((base_dir / f"dataloader/{args.dataset}.py").absolute())
dataset = datasets.load_dataset(dataloader_path, name=args.dataset_config, use_auth_token=True, )["train"]

# dataset = dataset.select(random.sample(list(np.arange(len(dataset))), 100))

label2id = {}
for tags in dataset["tags"] + dataset["tags_gold"]:
    for tag in tags:
        if tag not in label2id:
            label2id[tag] = len(label2id)
id2label = {v: k for k, v in label2id.items()}
dataset = dataset.map(partial(update_labels, ids=[]))


LARGE_NEG_NUMBER = -100000000
assert len(dataset) < abs(LARGE_NEG_NUMBER)
errors = np.array([i["tags"] != i["tags_gold"] for i in dataset])
is_labelled = np.zeros(len(dataset)).astype(bool)
args.out_dir.mkdir(exist_ok=True, parents=True)
final_ranks = np.zeros(len(dataset))
df_dict = defaultdict(list)
precisions = []
recalls = []
scorer_to_scores = {"aum": [], "dm": [], "cu": [], "aum_logits": []}
for i in trange(math.ceil(len(dataset) / args.k), desc="Outer loop"):
    error_scores, predictions = predict(dataset, label2id=label2id, id2label=id2label,
                     model_name=args.model_name, out_dir=args.out_dir, num_epochs=args.n_epochs,
                                        random_seed=args.seed)
    for scorer in scorer_to_scores:
        if args.aggregation == "both":
            scores = (error_scores[scorer + "_train"] + error_scores[scorer + "_test"]) / 2
        elif args.aggregation == "train":
            scores = error_scores[scorer + "_train"]
        elif args.aggregation == "test":
            scores = error_scores[scorer + "_test"]
        else:
            raise ValueError(args.aggregation)
        scorer_to_scores[scorer] = scores

    scores = scorer_to_scores[args.scorer]
    topk_indices = []
    for idx in np.argsort(scores)[::-1]:
        if not is_labelled[idx]:
            topk_indices.append(idx)
        if len(topk_indices) == args.k:
            break
    for idx in topk_indices:
        final_ranks[idx] = (final_ranks != 0).sum() + 1

    is_labelled[topk_indices] = True

    current_ranks = rankdata(-scores, method="ordinal")

    ranks = np.zeros_like(final_ranks)
    ranks[is_labelled] = final_ranks[is_labelled]
    ranks[~is_labelled] = current_ranks[~is_labelled] + len(ranks)

    ranked_texts = [None for _ in final_ranks]
    ranked_tags = [None for _ in final_ranks]
    ranked_true_tags = [None for _ in final_ranks]

    for i, rank in enumerate(final_ranks):
        if rank == 0:
            continue
        ranked_texts[int(rank)-1] = dataset["tokens"][i]
        ranked_tags[int(rank-1)] = dataset["tags"][i]
        ranked_true_tags[int(rank-1)] = dataset["tags_gold"][i]

    with open(args.out_dir / "errors.json", "w") as f:
        json.dump({"texts": ranked_texts, "tags": ranked_tags, "true_tags": ranked_true_tags}, f)

    rank_scores = ranks.copy()
    rank_scores[rank_scores == 0] = ranks.sum()
    rank_scores = - rank_scores

    df_dict["ap"].append(average_precision_score(errors, rank_scores))
    df_dict["labeled_ratio"].append(np.mean([i["tags"] == i["tags_gold"] for i in dataset]))
    df_dict["topk_error_rate"].append(np.mean(errors[topk_indices]))

    for scorer in scorer_to_scores:
        df_dict["ap_" + scorer].append(average_precision_score(errors, scorer_to_scores[scorer]))

    p, r, _ = precision_recall_curve(errors, rank_scores)
    precisions.append(p)
    recalls.append(r)

    labelled_ids = np.array(dataset["id"])[is_labelled]
    dataset = dataset.map(partial(update_labels, ids=labelled_ids))

    df = pd.DataFrame(df_dict)
    df.to_csv(args.out_dir / "results.csv")

    np.save(args.out_dir / "precisions.npy", precisions)
    np.save(args.out_dir / "recalls.npy", recalls)

    if args.no_active:
        break
