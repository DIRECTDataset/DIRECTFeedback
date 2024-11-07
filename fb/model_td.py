"""Model evaluation pipeline."""
import json

import evaluate
from nltk.tokenize import word_tokenize
import pytorch_lightning as pl
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

from fb.utils_td import FH


BLEU = evaluate.load("sacrebleu")
ROUGE = evaluate.load("rouge")
METEOR = evaluate.load("meteor")
BERTSCORE = evaluate.load("bertscore")
FHSCORE = FH()


class FeedbackModel(pl.LightningModule):
    """Model for Reading Comprehension Feedback Generation."""
    tokenizer = T5Tokenizer.from_pretrained(
        "t5-base"
    )

    def __init__(self, model_name="t5-base"):
        super().__init__()

        # Load the model configuration
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name
        )

        with open("data/article-id_mapping.json", encoding="utf-8") as json_in:
            self.mapping = json.load(json_in)

        # allocator variables for validation loop
        self._val_ref, self._val_ref_tok = [], []
        self._val_out, self._val_out_tok = [], []
        self._overall_fh = []

        self.save_hyperparameters()

    def forward(self, batch, **generation_kwargs):
        """Generate human-readable text from model."""
        if not generation_kwargs:
            generation_kwargs = {
                "max_new_tokens": 256,
                "num_beams": 5,
                "early_stopping": True,
                "length_penalty": 2.0
            }

        output_seq = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            **generation_kwargs
        )
        output_text = self.tokenizer.batch_decode(
            output_seq,
            skip_special_tokens=True
        )
        return output_text

    def on_validation_epoch_start(self):
        self.model.eval()
        torch.set_grad_enabled(False)

        torch.cuda.empty_cache()

    def validation_step(self, batch, batch_idx):
        """Prepare output and reference solutions for evaluation."""

        # create and collect references and candidate generations
        reference = self.tokenizer.batch_decode(
            batch["labels"],
            skip_special_tokens=True
        )
        out = self(batch)

        self._val_ref.extend(
            reference
        )
        self._val_out.extend(
            out
        )
        self._val_ref_tok.extend(
            " ".join(word_tokenize(r)) for r in reference
        )
        self._val_out_tok.extend(
            " ".join(word_tokenize(c)) for c in out
        )

        # log output
        if batch_idx < 50:
            for o, r in zip(out, reference):
                print("out:", o, "\tref:", r)

        with open("outputs.txt", "a", encoding="utf-8") as file_out:
            for i, (o, r) in enumerate(zip(out, reference)):
                file_out.write("\t".join([
                    str(batch['dataset'][i].item()),
                    str(batch['article_id'][i].item()),
                    str(batch['question_id'][i].item()),
                    o,
                    r
                ]) + "\n")

        print("Generations have been saved to output.txt")

        # compute helpfulness
        for i in range(len(reference)):
            _, info, info_H, info_L, truth = FHSCORE(
                reference[i], out[i], batch["article_id"][i].item(),
                multirc_onset_ids=batch["multirc"][i]
            )
            self._overall_fh.append((info, info_H, info_L, truth))

    def on_validation_epoch_end(self):
        """Calculate and log validation set metrics."""
        results = {}

        # traditional metrics
        for metric, metric_name, index in [
            (BLEU, "BLEU", "score"),
            (ROUGE, "ROUGE", "rougeL"),
            (METEOR, "METEOR", "meteor"),
            (BERTSCORE, "BERTSCORE", "f1")
        ]:
            if metric_name in {"ROUGE", "METEOR"}:
                score = getattr(metric, "compute")(
                    predictions=self._val_out_tok,
                    references=self._val_ref_tok,
                )[index]
            else:
                score = getattr(metric, "compute")(
                    predictions=self._val_out,
                    references=self._val_ref,
                    **({
                        "model_type": "microsoft/deberta-xlarge-mnli",
                        "lang": "en",
                        "rescale_with_baseline": True
                    } if metric_name == "BERTSCORE" else {})
                )[index]

            if isinstance(score, list):
                score = sum(score)/len(score)

            results[metric_name] = score
            self.log(metric_name, score, sync_dist=True)

        # proposed metrics
        for name, metric in zip(
            ["info", "info_H", "info_L", "truth"],
            zip(*self._overall_fh)
        ):
            if name in {"info_H", "info_L"}:
                print(name, "before:", len(metric))
                metric = [value for value in metric if value != 1]
                print(name, "after:", len(metric))
            score = sum(metric)/len(metric) if metric else 1

            results[name] = score
            self.log(name, score, on_epoch=True, sync_dist=True)

        self._val_out_tok.clear()
        self._val_ref_tok.clear()
        self._val_out.clear()
        self._val_ref.clear()

        self._overall_fh.clear()
