from typing import Any

import evals
import evals.metrics
from evals.api import CompletionFn
from evals.prompt.base import is_chat_prompt


class Translate(evals.Eval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        samples_jsonl: str,
        *args,
        max_tokens: int = 500,
        num_few_shot: int = 0,
        few_shot_jsonl: str = None,
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        assert len(completion_fns) == 1, "Translate only supports one completion fn"
        self.max_tokens = max_tokens
        self.samples_jsonl = samples_jsonl

    def eval_sample(self, sample: Any, *_):
        prompt = sample["input"]
        expected = sample["ideal"]

        if isinstance(expected, tuple):
            expected = list(expected)
        elif not isinstance(expected, list):
            expected = [expected]

        result = self.completion_fn(
            prompt=prompt,
            max_tokens=self.max_tokens,
        )
        sampled = result.get_completions()[0]

        score = None
        if expected is not None:
            score = self.bleu.sentence_score(sampled, expected).score
            evals.record.record_metrics(sacrebleu_sentence_score=score)

            match = score > 30

            if score is not None:
                evals.record.record_match(
                    match, expected=expected, sampled=sampled, sacrebleu_sentence_score=score
                )
            return match

    def run(self, recorder):
        samples = self.get_samples()
        self.eval_all_samples(recorder, samples)
        events = recorder.get_events("match")

        sampled = list(map(lambda e: e.data["sampled"], events))
        expected = list(map(lambda e: e.data["expected"], events))
        sacrebleu_score = BLEU().corpus_score(sampled, [expected]).score

        return {
            "accuracy": evals.metrics.get_accuracy(events),
            "sacrebleu_score": sacrebleu_score,
        }
