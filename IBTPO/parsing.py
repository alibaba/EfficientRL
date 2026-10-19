"""Token-level reasoning-step parsing utilities."""

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ParsedStep:
    token_ids: List[int]
    logprobs: List[float]

    @property
    def mean_logprob(self) -> float:
        if not self.logprobs:
            return -math.inf
        clipped = [min(0.0, max(-2.0, value)) for value in self.logprobs]
        return sum(clipped) / len(clipped)


def _find_first_pattern(
    sequence: Sequence[int],
    patterns: Sequence[Sequence[int]],
    start: int,
) -> Tuple[int, Sequence[int]]:
    best_index = -1
    best_pattern: Sequence[int] = ()
    for pattern in patterns:
        if not pattern:
            continue
        limit = len(sequence) - len(pattern) + 1
        for index in range(max(0, start), max(0, limit)):
            if list(sequence[index:index + len(pattern)]) == list(pattern):
                if best_index < 0 or index < best_index or (
                    index == best_index and len(pattern) > len(best_pattern)
                ):
                    best_index = index
                    best_pattern = pattern
                break
    return best_index, best_pattern


def split_reasoning_steps(
    token_ids: Sequence[int],
    logprobs: Sequence[float],
    *,
    max_steps: int,
    end_think_ids: Sequence[int],
    separator_ids: Sequence[Sequence[int]],
    min_step_tokens: int,
    max_step_tokens: Optional[int],
) -> List[ParsedStep]:
    """Split one completion without ever dropping or reordering tokens.

    Separators are retained at the end of each reasoning step. If the natural
    number of steps exceeds ``max_steps``, the tail is merged into the final
    step. This also preserves the thinking text when only one depth slot is
    available, which the original parser did not guarantee.
    """
    if len(token_ids) != len(logprobs):
        raise ValueError("token_ids and logprobs must have equal lengths")
    if max_steps <= 0 or not token_ids:
        return []

    ids = list(token_ids)
    lps = list(logprobs)
    think_end, matched_end = _find_first_pattern(ids, [end_think_ids], 0)
    if think_end >= 0:
        boundary = think_end + len(matched_end)
        think_ids, answer_ids = ids[:boundary], ids[boundary:]
        think_lps, answer_lps = lps[:boundary], lps[boundary:]
    else:
        think_ids, answer_ids = ids, []
        think_lps, answer_lps = lps, []

    chunks: List[ParsedStep] = []
    cursor = 0
    while cursor < len(think_ids):
        search_from = min(len(think_ids), cursor + max(0, min_step_tokens))
        split_index, separator = _find_first_pattern(think_ids, separator_ids, search_from)
        separator_end = split_index + len(separator) if split_index >= 0 else len(think_ids)
        length_end = (
            min(len(think_ids), cursor + max_step_tokens)
            if max_step_tokens is not None
            else len(think_ids)
        )
        end = min(separator_end, length_end)
        if end <= cursor:
            end = min(len(think_ids), cursor + 1)
        chunks.append(ParsedStep(think_ids[cursor:end], think_lps[cursor:end]))
        cursor = end

    if answer_ids:
        chunks.append(ParsedStep(answer_ids, answer_lps))

    if len(chunks) > max_steps:
        kept = chunks[:max_steps - 1]
        tail = chunks[max_steps - 1:]
        merged_ids = [token for chunk in tail for token in chunk.token_ids]
        merged_lps = [value for chunk in tail for value in chunk.logprobs]
        chunks = kept + [ParsedStep(merged_ids, merged_lps)]

    reconstructed_ids = [token for chunk in chunks for token in chunk.token_ids]
    reconstructed_lps = [value for chunk in chunks for value in chunk.logprobs]
    if reconstructed_ids != ids or reconstructed_lps != lps:
        raise RuntimeError("step parsing changed the generated token stream")
    return chunks
