import json
import math
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Union, Optional, List, Dict
from collections import Counter, defaultdict

import datasets
import pandas as pd
import seqeval
from scipy.stats import rankdata
from sklearn.metrics import precision_score, recall_score, accuracy_score, average_precision_score, f1_score
from sklearn.model_selection import KFold, train_test_split
import transformers
import argparse
import torch
from tqdm import tqdm, trange
from transformers import PreTrainedTokenizerBase
import numpy as np
import evaluate
from transformers import DataCollatorWithPadding
from pytorch_lightning import seed_everything
from transformers.utils import PaddingStrategy


base_dir = Path(__file__).parent


@dataclass
class DataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        batch["position_ids"] =  torch.stack([torch.arange(sequence_length) for _ in batch["input_ids"]])
        return batch

def score_aum(outputs):
    scores = np.zeros(len(outputs[0].predictions))
    for output in outputs:
        for i_example,  (logits, label) in enumerate(zip(output.predictions, output.label_ids)):
            probs = torch.softmax(torch.tensor(logits).float(), dim=0)
            prob_true = probs[label].item()
            probs[label] = 0
            prob_max_other = max(probs)
            scores[i_example] += (prob_max_other - prob_true)

    return scores

def score_aum_logits(outputs):
    scores = np.zeros(len(outputs[0].predictions))
    for output in outputs:
        for i_example,  (logits, label) in enumerate(zip(output.predictions, output.label_ids)):
            logits = torch.tensor(logits).float()
            prob_true = logits[label].item()
            logits[label] = 0
            prob_max_other = max(logits)
            scores[i_example] += (prob_max_other - prob_true)

    return scores

def score_dm(outputs):
    scores = np.zeros(len(outputs[0].predictions))
    for output in outputs:
        for i_example,  (logits, label) in enumerate(zip(output.predictions, output.label_ids)):
            probs = torch.softmax(torch.tensor(logits).float(), dim=0)
            prob_true = probs[label].item()
            scores[i_example] += prob_true

    return (1 - scores)  # higher -> more likely to be error

def score_cu(outputs):
    scores = np.zeros(len(outputs[0].predictions))
    output = outputs[-1]
    for i_example,  (logits, label) in enumerate(zip(output.predictions, output.label_ids)):
        probs = torch.softmax(torch.tensor(logits).float(), dim=0)
        prob_true = probs[label].item()
        scores[i_example] = prob_true

    return 1 - scores # higher -> more likely to be error

def score_retag(outputs):
    scores = np.zeros(len(outputs[0].predictions))
    output = outputs[-1]
    for i_example,  (logits, label) in enumerate(zip(output.predictions, output.label_ids)):
        probs = torch.softmax(torch.tensor(logits).float(), dim=0)
        prob_true = probs[label].item()
        probs[label] = 0
        prob_max_other = max(probs)
        scores[i_example] = (prob_max_other - prob_true)

    return scores


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


def predict(dataset, label2id, id2label, model_name, out_dir, n_epochs):

    cv = KFold(10, shuffle=True, random_state=42)

    error_scores = {
        "aum_train": np.zeros(len(dataset)),
        "aum_test": np.zeros(len(dataset)),

        "dm_train": np.zeros(len(dataset)),
        "dm_test": np.zeros(len(dataset)),
        "retag_train": np.zeros(len(dataset)),
        "retag_test": np.zeros(len(dataset)),
        "cu_train": np.zeros(len(dataset)),
        "cu_test": np.zeros(len(dataset)),
    }
    preds = None

    for train_index, test_index in tqdm(list(cv.split(dataset["text"])), desc="CV"):
        model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name,
                                                                             num_labels=len(label2id),
                                                                             label2id=label2id,
                                                                             id2label=id2label)

        train_dataset = dataset.select(train_index)
        test_dataset = dataset.select(test_index)

        train_dataset = train_dataset.map(lambda x: tokenizer(x["text"], truncation=True, max_length=512),
                                                       batched=True,
                                                       remove_columns=["text"])
        test_dataset = test_dataset.map(lambda x: tokenizer(x["text"], truncation=True, max_length=512),
                                        batched=True, remove_columns=["text"])

        logits_getter = LogitsGetterCallback(train_data=train_dataset, test_data=test_dataset)

        if torch.cuda.is_available():
            batch_size_denominator = max(1, torch.cuda.device_count()) # Make batch size consistent across computation environments with different numbers of GPUs
        else:
            batch_size_denominator = 1
        batch_size = 64
        num_epochs = n_epochs

        training_args = transformers.TrainingArguments(
            output_dir=out_dir,
            learning_rate=5e-5,
            per_device_train_batch_size=batch_size // batch_size_denominator,
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
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            callbacks=[logits_getter]
        )
        logits_getter.trainer = trainer

        trainer.train()

        outputs = trainer.predict(test_dataset)

        error_scores["aum_train"][train_index] += score_aum_logits(logits_getter.train_outputs)
        error_scores["dm_train"][train_index] += score_dm(logits_getter.train_outputs)
        error_scores["cu_train"][train_index] += score_cu(logits_getter.train_outputs)
        error_scores["retag_train"][train_index] += score_retag(logits_getter.train_outputs)

        error_scores["aum_test"][test_index] = score_aum_logits(logits_getter.test_outputs + [outputs])
        error_scores["dm_test"][test_index] = score_dm(logits_getter.test_outputs + [outputs])
        error_scores["cu_test"][test_index] = score_cu(logits_getter.test_outputs + [outputs])
        error_scores["retag_test"][test_index] = score_retag(logits_getter.test_outputs + [outputs])

        if preds is None:
            preds = np.zeros((len(dataset), ) + outputs.predictions.shape[1:])
        preds[test_index] = outputs.predictions


    return error_scores, preds


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)

    precision = precision_score(labels, predictions, average="micro")
    recall = recall_score(labels, predictions, average="micro")
    f1 = f1_score(labels, predictions, average="micro")
    accuracy = accuracy_score(labels, predictions)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def update_labels(example, ids):
    if example["id"] in ids:
        if example["error"] > 0:
            example["label"] = example["true_label"]

    return example


seqeval = evaluate.load("seqeval")

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--scorer", required=True)
parser.add_argument("--out_dir", required=True, type=Path)
parser.add_argument("--n_epochs", required=True, type=int)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--model_name", default="distilroberta-base")
parser.add_argument("--aggregation", choices=["train", "test", "both"], default="both")
parser.add_argument("--no_active", action="store_true")
parser.add_argument("--k", default=50, type=int)
parser.add_argument("--dataset_config")
args = parser.parse_args()

seed_everything(args.seed)

tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name, add_prefix_space=True)

dataloader_path = str((base_dir / f"dataloader/{args.dataset}.py").absolute())
dataset = datasets.load_dataset(dataloader_path, name=args.dataset_config, use_auth_token=True, )["train"]

label2id = {l: i for i, l in enumerate(sorted(set(dataset['label'])))}
id2label = {v: k for k, v in label2id.items()}


def preprocess(example):
    example['label'] = label2id[example['label']]
    example["true_label"] = label2id[example["true_label"]]
    example["error"] = example["label"] != example["true_label"]

    return example

dataset = dataset.map(preprocess)

# error_indices = sorted(np.where(dataset["error"])[0])
# fillers = sorted(random.sample(list(np.where(~np.array(dataset["error"]))[0]), 2000))
# dataset = dataset.select(error_indices + fillers)


LARGE_NEG_NUMBER = -100000000
assert len(dataset) < abs(LARGE_NEG_NUMBER)
is_labelled = np.zeros(len(dataset)).astype(bool)
args.out_dir.mkdir(exist_ok=True, parents=True)
errors = np.array(dataset['error'])
final_ranks = np.zeros(len(dataset))
df_dict = defaultdict(list)
previous_scores = []
scorer_to_scores = {"aum": [], "dm": [], "cu": []}
for i in trange(math.ceil(len(dataset) / args.k), desc="Outer loop"):
    error_scores, predictions = predict(dataset, label2id=label2id, id2label=id2label,
                     model_name=args.model_name, out_dir=args.out_dir, n_epochs=args.n_epochs)

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
    current_ranks = rankdata(-scores)

    ranks = np.zeros_like(final_ranks)
    ranks[is_labelled] = final_ranks[is_labelled]
    ranks[~is_labelled] = current_ranks[~is_labelled] + len(ranks)

    ranked_texts = [None for _ in final_ranks]
    ranked_labels = [None for _ in final_ranks]
    ranked_true_labels = [None for _ in final_ranks]

    for i, rank in enumerate(final_ranks):
        if rank == 0:
            continue
        ranked_texts[int(rank)-1] = dataset["text"][i]
        ranked_labels[int(rank-1)] = id2label[dataset["label"][i]]
        ranked_true_labels[int(rank-1)] = id2label[dataset["true_label"][i]]

    with open(args.out_dir / "errors.json", "w") as f:
        json.dump({"texts": ranked_texts, "labels": ranked_labels, "true_labels": ranked_true_labels}, f)

    rank_scores = ranks.copy()
    rank_scores[rank_scores == 0] = ranks.sum()
    rank_scores = - rank_scores

    df_dict["ap"].append(average_precision_score(errors, rank_scores))
    df_dict["labeled_ratio"].append(np.mean([i["label"] == i["true_label"] for i in dataset]))
    df_dict["topk_error_rate"].append(np.mean(errors[topk_indices]))

    for scorer in scorer_to_scores:
        df_dict["ap_" + scorer].append(average_precision_score(errors, scorer_to_scores[scorer]))

    # df_dict["final_ap"].append(average_precision_score(errors, rank_scores))
    # error_indices = np.arange(len(dataset))[errors.astype(bool)]
    # labelled_error_indices = np.arange(len(dataset))[is_labelled & errors.astype(bool)]

    # correction_metrics = compute_metrics((predictions[errors.astype(bool)],
    #                                             dataset.select(error_indices)["label"]))
    # for metric, value in correction_metrics.items():
    #     df_dict["correction_" + metric] = value

    # final_correction_metrics = compute_metrics((predictions[is_labelled & errors.astype(bool)],
    #                                       dataset.select(labelled_error_indices)["label"]))
    #
    # for metric, value in final_correction_metrics.items():
    #     df_dict["final_correction_" + metric] = value

    labelled_ids = np.array(dataset["id"])[is_labelled]
    dataset = dataset.map(partial(update_labels, ids=labelled_ids))

    df = pd.DataFrame(df_dict)
    df.to_csv(args.out_dir / "results.csv")

    if args.no_active:
        break
