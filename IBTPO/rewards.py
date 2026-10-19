"""Reward functions used by the public IBTree demo."""

from __future__ import annotations

import re
from typing import Optional


class MathRewardVerifier:
    """Binary mathematical-answer verifier backed by ``math_verify``."""

    @staticmethod
    def _answer_text(text: str) -> tuple[str, bool]:
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        return (match.group(1).strip(), True) if match else (text, False)

    def __call__(self, completion: str, solution: Optional[str]) -> float:
        if not solution:
            return 0.0
        try:
            from latex2sympy2_extended import NormalizationConfig
            from math_verify import LatexExtractionConfig, parse, verify

            completion_text, has_answer_tag = self._answer_text(completion)
            solution_text, _ = self._answer_text(solution)
            gold = parse(solution_text, extraction_mode="first_match")
            if not gold:
                return 0.0
            if has_answer_tag:
                prediction = parse(completion_text, extraction_mode="first_match")
            else:
                prediction = parse(
                    completion_text,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                boxed=True,
                                units=True,
                            ),
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode="first_match",
                )
            return float(verify(gold, prediction))
        except (ImportError, ModuleNotFoundError) as error:
            raise RuntimeError(
                "Math verification requires math_verify and latex2sympy2_extended"
            ) from error
        except Exception:
            return 0.0
