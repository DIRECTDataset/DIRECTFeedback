"""Interface for supervised-finetuning of feedback model."""
import argparse
import sys

import pytorch_lightning as pl
import torch
from transformers import DataCollatorForSeq2Seq
import yaml

from fb.dataloader_td import FeedbackDataset
from fb.model_td import FeedbackModel


def train(cfg, ckpt_load, test):
    # prepare trainer
    trainer = pl.Trainer(
        **cfg["trainer"],
        devices=1 if test else "auto",
        strategy="ddp_find_unused_parameters_false"
    )

    print("Load data ...")

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=FeedbackModel.tokenizer
    )

    test_dataset = FeedbackDataset.load_from_confg(
        cfg["data"],
        split="test",
    )
    test_data = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=1,
        collate_fn=data_collator,
        drop_last=False
    )

    print("COMPLETE")
    print("Build model ...")

    model = FeedbackModel(ckpt_load)

    print("COMPLETE")

    print("Start test ...")
    trainer.validate(
        model,
        dataloaders=test_data
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interface to train a Feedback Generation Model.",
        epilog="The following arguments are required: CONFIG"
    )

    parser.add_argument(
        "config", metavar="CONFIG", type=argparse.FileType("r"),
        help="YAML-file holding all data and model parameters"
    )
    parser.add_argument(
        "--load", metavar="CKPT_NAME", default=None,
        help="path to trained model state"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="whether to run in train or test mode"
    )

    if len(sys.argv) == 1:
        parser.print_help()
    else:
        args = parser.parse_args()
        cfg_dict = yaml.load(args.config, Loader=yaml.FullLoader)

        train(cfg_dict, args.load, True)

