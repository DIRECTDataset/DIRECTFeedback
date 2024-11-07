"""Data preparation & preprocessing."""
import json
import re

import torch
from torch.utils.data import Dataset

from fb.model_td import FeedbackModel


class FeedbackDataset(Dataset):
    """Offers randomized access and preprocessing of data items.

    Attributes:
        mapping (dict):
            Mapping from article identifiers (str)
            to the text of the corresponing article (str).
        data (list):
            Sequence of data items, each item being a 2-item tuple,
            holding an article identifier and an item belonging to it:
            (
                0: article_id
            ),
            (
                0: question
                1: key_sentence
                2: correct_answer
                3: wrong_answer
                4: feedback
            )
        data_confg (dict):
            Parameters for text preprocessing.
        split (str):
            One of {'train', 'valid', 'test'}. Defaults to 'train'.

    """
    field_names = {
        "question": 0, "key_sentence": 1,
        "correct_answer": 2, "wrong_answer": 3,
        "feedback": 4, "article": 5
    }
    chunking_re = re.compile(
        r'(?:(?<=[!?"])|(?=")|\n| {4}|</s>)|(?:(?<=[,.;:])(?=\D)) *'
    )

    def __init__(self, split):
        self.mapping = {}
        self.data = []

        self.data_confg = {}
        self.split = split

    @classmethod
    def load_from_confg(cls, data_confg, split="train"):
        """Instantiate class from a collection of file paths.

        Args:
            data_confg (dict):
                Mapping of file paths, of the following format,
                and parameters for text processing.
                See data/ for example files.
                {
                    "mapping": <.json>,
                    "path": <.csv with \t separator>,
                    "split":
                    {
                        "val": <.txt with one identifier per line>,
                        "test": <.txt with one identifier per line>
                    }
                    ...
                }
            split (Optional[str]):
                One of {'train', 'valid', 'test'}. Defaults to 'train'.
            sub_include (Optional[callable]):
                Function that accepts the dataset name and item id
                as input and returns True if this item is to be
                included in the data for sub task training.
                Use only when self.data_confg[mode]='multi-task'.

        Returns:
            B
            cls: Instance of the dataset class.

        """
        if split not in {"train", "valid", "test", "rf"}:
            raise ValueError(
                "Split needs to be one of: {'train', 'valid', 'test', 'rf'}"
            )

        data_split = set()
        # for the "train" split we will later check for the absence of the
        # identifier in the combined split of "test" and "valid"
        # for "test" and "valid" we will check for the existence
        if split in {"train", "test"}:
            with open(data_confg["split"]["test"], encoding="utf-8") as test_in:
                data_split.update({line.strip() for line in test_in})
        if split in {"train", "valid"}:
            with open(data_confg["split"]["val"], encoding="utf-8") as valid_in:
                data_split.update({line.strip() for line in valid_in})
        if split in {"train", "rf"} and "rf" in data_confg["split"]:
            with open(data_confg["split"]["rf"], encoding="utf-8") as rf_in:
                data_split.update({line.strip() for line in rf_in})

        # create class instance
        obj = cls(split)
        obj.data_confg = data_confg

        d = {}
        d2 = {}
        with open(data_confg["mapping"], encoding="utf-8") as json_in:
            obj.mapping = json.load(json_in)
        with open(data_confg["path"], encoding="utf-8") as file_in:
            file_in.readline()
            for line in file_in:
                dataset, article_id, question_id, *item = line.strip().split("\t")
                dataset = {"DIRECT": 0, "DIRECT-Feedback": 1}[dataset]

                if split == "train" and article_id not in data_split:
                    obj.data.append((dataset, article_id, question_id, item))

                elif split != "train" and article_id in data_split:
                    obj.data.append((dataset, article_id, question_id, item))

        return obj

    def __getitem__(self, idx):
        """Data retrieval and preprocessing."""
        # retrieve item
        dataset, article_id, question_id, item = self.data[idx]

        item.append(self.mapping[article_id])
        item_dict = {attr: item[i] for attr, i in self.field_names.items()}

        input_data = self._prepare_input(
            item_dict
        )

        return {
            #"gold_answer": item_dict["correct_answer"],
            **input_data,
            **self._prepare_output(item_dict),
            **({"dataset": int(dataset)}),
            **({"article_id": int(article_id)}),
            **({"question_id": int(question_id)}),
            **(self._prepare_evaluation(item_dict, article_id)),
        }

    def _prepare_input(self, item, input_include=None):
        """Prepare input token ids and attention mask."""
        # prepare input components by selection, prefixing and concatenation
        input_ = []
        for attr in (input_include or self.data_confg["input_include"]):
            # retrieve partial components
            text = item[attr]
            # add special prefixes to input segments
            input_.append(
                self.data_confg.get("prefixes", {}).get(attr, "") +
                text
            )
        # join input together
        input_str = "".join(input_)

        # encode source text as input
        encoding = FeedbackModel.tokenizer(
            input_str,
            max_length=self.data_confg["max_source_length"],
            truncation=True,
            return_tensors="pt",
        )
        input_ids, att_mask = encoding.input_ids[0], encoding.attention_mask[0]

        if len(input_ids) == self.data_confg["max_source_length"]:
            # if something needs to be truncated it should be the article first
            # only if it still not fits the rest should be truncuated
            new_enc = FeedbackModel.tokenizer(
                "".join(input_[1:]),
                max_length=self.data_confg["max_source_length"],
                truncation=True,
                return_tensors="pt",
                return_length=True
            )
            input_ids[-new_enc.length:] = new_enc.input_ids[0, -new_enc.length:]
            att_mask[-new_enc.length:] = new_enc.attention_mask[0, -new_enc.length:]

        if len(input_ids) >= self.data_confg["max_source_length"]:
            print(len(input_ids))

        return {"input_ids": input_ids, "attention_mask": att_mask}

    def _prepare_output(self, item):
        """Prepare target labels and loss mask."""
        # for randomized multitask training, select a task
        task = self.data_confg["output_include"]
        output_str = item[task]

        # encode target text as output
        target_encoding = FeedbackModel.tokenizer(
            output_str,
            max_length=self.data_confg["max_target_length"],
            truncation=True,
            return_tensors="pt"
        )
        labels = target_encoding.input_ids[0]

        return {"labels": labels}


    def _prepare_evaluation(self, item, article_id):
        """Prepare auxiliary task data used for evaluation."""

        multirc_format = "multirc question: {} answer: {} paragraph: "
        multirc_encoding = FeedbackModel.tokenizer(
            multirc_format.format(
                item["question"],
                item["correct_answer"]
            ),
            max_length=256,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return {"multirc": multirc_encoding.input_ids[0]}

    def __iter__(self):
        """Returns a sorted iterarion over the data set."""
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        """Returns the number of items in the data set."""
        return len(self.data)
