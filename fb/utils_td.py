"""Proposed evaluation measure."""
import json

from nltk.tokenize import sent_tokenize
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration


class FH:
    """Feedback helpfulness metric.

    Attributes:
        model (Seq2Seq Model):
            A model pretrained on the MultiRC  dataset to evaluate
            the correctness of an answer to a question, given a
            multi-sentence paragraph.
        tokenizer (Tokenizer):
            Respective tokenizer of the above model.

    """
    def __init__(self, device="auto"):
        self.tokenizer = T5Tokenizer.from_pretrained(
            "t5-large", device_map=device
        )
        self.model = T5ForConditionalGeneration.from_pretrained(
            "t5-large", device_map=device
        )
        self.model.eval()

        self._soft_max = torch.nn.Softmax(dim=2)
        self._expected_bos_token = self.tokenizer(
            "True",
            padding=True,
            return_tensors="pt"
        ).input_ids[:, 0]

        with open("data/article-id_mapping.json", encoding="utf-8") as json_in:
            self.mapping = json.load(json_in)

    def __call__(
        self,
        gold_feedback, model_feedback, article_id,
        question=None, gold_answer=None,
        multirc_onset_ids=None
    ):
        """Compute metric score.

        Arguments:
            gold_feedback (str):
                Feedback deemed as good.
            model_feedback (str):
                Feedback produced by a model.
            article_id (str/int):
                Identifier of corresponding text.
            question (str):
                Question the student had to answer.
            gold_answer (str):
                The correct answer to this question.
            multirc_onset_ids (tensor):
                1D Tensor with holding the precomputable
                onset of the multirc task in the form:
                "multirc question: <question>
                answer: <gold answer> paragraph:"

        Returns:
            tuple: overall, informativenes, truthfulness scores

        """
        if multirc_onset_ids is None and None in {
            question, gold_answer
        }:
            raise ValueError(
                "Need to specify either onset_ids or "
                "all of question, gold_answer, student_answer!"
            )

        multirc_format = "multirc question: {} answer: {} paragraph: "
        mnli_format = "mnli hypothesis: {} premise: {}"
        if multirc_onset_ids is None:
            onset_encoding = self.tokenizer(
                [
                    multirc_format.format(
                        question, gold_answer
                    ),
                    mnli_format.format(
                        model_feedback,
                        self.mapping[str(article_id)]
                    )
                ],
                padding="longest",
                return_tensors="pt"
            )
            multirc_onset_ids = onset_encoding.input_ids[0]
            mnli_ids = onset_encoding.input_ids[1:]
        else:
            onset_encoding = self.tokenizer(
                mnli_format.format(
                    model_feedback,
                    self.mapping[str(article_id)]
                ),
                padding="longest",
                return_tensors="pt"
            )
            mnli_ids = onset_encoding.input_ids

        # compute the alignment between text and feedback
        contr, entail = self._feedback_answer_alignment(
            inp=mnli_ids.to(self.model.device),
            expected_answer=["contradiction", "entailment"]
        )

        truthfulness = (1 + min(entail, 0.5) - contr)/1.5

        # prepare multirc input format
        # consisting of a precomputable onset
        # ... and a sentence tokenized suffix holding the feedback
        gold_feedback = "".join(
            f"<b>Sent {i}: </b>{s}<br>" for i, s in enumerate(
                sent_tokenize(gold_feedback), 1
            )
        )
        model_feedback = "".join(
            f"<b>Sent {i}: </b>{s}<br>" for i, s in enumerate(
                sent_tokenize(model_feedback), 1
            )
        )
        feedback_encoding = self.tokenizer(
            [gold_feedback, model_feedback],
            padding="longest",
            return_tensors="pt"
        )
        feedback_ids = feedback_encoding.input_ids

        # prepare the concatenation of those two
        additional_pad = torch.zeros(
            feedback_ids.size()[1],
            dtype=multirc_onset_ids.dtype,
            device=multirc_onset_ids.device
        )
        multirc_onset_ids = torch.cat(
            (multirc_onset_ids, additional_pad), 0
        )

        # use multirc task to
        # compute the alignment between gold answer and feedback
        a, b = self._feedback_answer_alignment(
            multirc_onset_ids.to(self.model.device),
            feedback_ids[0],
            feedback_ids[1]
        )

        informativeness = 1 - abs(a-b)
        excessive_informativeness = 1 - max(b-a, 0)
        absent_informativeness = 1 - max(a-b, 0)

        overall = (
            0.75*informativeness +
            0.25*truthfulness
        )

        return (
            overall,
            informativeness,
            excessive_informativeness,
            absent_informativeness,
            truthfulness
        )


    def _feedback_answer_alignment(
        self,
        onset=None,
        gold_feedback=None,
        pred_feedback=None,
        inp=None,
        expected_answer=None
    ):
        """Compute feedback and answer alignment."""
        if inp is None and None in {onset, gold_feedback, pred_feedback}:
            raise ValueError(
                "Need to specify either inp or "
                "all of onset, gold_feedback, pred_feedbac!"
            )

        if inp is None:
            onset_end = (onset == 1).nonzero()  # padding start

            # merge the precomputed onset with gold and generated feedback
            inp = onset.repeat(2, 1)

            inp[0, onset_end:onset_end+len(gold_feedback)] = gold_feedback
            inp[1, onset_end:onset_end+len(pred_feedback)] = pred_feedback

        attention_mask = torch.zeros(inp.size(), device=inp.device)
        attention_mask[inp != 0] = 1

        # we are interested in the probability assigned to True
        # this tells us how likely the answer is with respect to the feedback
        if expected_answer is None:
            expected_bos_token = self._expected_bos_token
        else:
            expected_bos_token = self.tokenizer(
                expected_answer,
                padding=True,
                return_tensors="pt"
            ).input_ids[:, 0]

        out = self.model(
            input_ids=inp,
            attention_mask=attention_mask,
            labels=torch.ones(
                (len(inp), 1),
                device=inp.device,
                dtype=inp.dtype
            )
        )

        true_prob = self._soft_max(
            out.logits
        )[:, 0, expected_bos_token].flatten().tolist()

        return true_prob
