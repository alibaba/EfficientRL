"""Command-line entry point for the standalone IBTree demo."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from opensource_ibtpo.config import CONFIG, DEFAULT_PROMPTS, DEFAULT_SOLUTIONS
    from opensource_ibtpo.sampler import IBTreeSampler
else:
    from .config import CONFIG, DEFAULT_PROMPTS, DEFAULT_SOLUTIONS
    from .sampler import IBTreeSampler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run standalone IBTree sampling with vLLM")
    parser.add_argument("--model", default=CONFIG.engine.model)
    parser.add_argument(
        "--prompt",
        dest="prompts",
        action="append",
        help="Prompt to sample; repeat this option for multiple prompts",
    )
    parser.add_argument(
        "--solution",
        dest="solutions",
        action="append",
        help="Reference solution aligned with each --prompt",
    )
    parser.add_argument("--output", default=CONFIG.output_path)
    parser.add_argument("--tensor-parallel-size", type=int, default=CONFIG.engine.tensor_parallel_size)
    parser.add_argument("--target-trajectories", type=int, default=CONFIG.tree.target_trajectories)
    parser.add_argument("--temperature", type=float, default=CONFIG.generation.temperature)
    parser.add_argument("--top-p", type=float, default=CONFIG.generation.top_p)
    parser.add_argument("--top-k", type=int, default=CONFIG.generation.top_k)
    parser.add_argument("--max-tokens", type=int, default=CONFIG.generation.max_tokens)
    parser.add_argument("--seed", type=int, default=CONFIG.generation.seed)
    parser.add_argument("--show-text", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    problems = list(args.prompts) if args.prompts is not None else list(DEFAULT_PROMPTS)
    if args.solutions is not None:
        solutions = list(args.solutions)
    elif args.prompts is None:
        solutions = list(DEFAULT_SOLUTIONS)
    else:
        solutions = [None] * len(problems)
    if len(solutions) != len(problems):
        parser.error("the number of --solution values must match --prompt values")
    for index, solution in enumerate(solutions):
        if solution is None:
            print(
                f"Warning: prompt {index} has no reference solution; correctness "
                "rewards will use the configured reward floor."
            )

    config = replace(
        CONFIG,
        engine=replace(
            CONFIG.engine,
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
        ),
        generation=replace(
            CONFIG.generation,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            seed=args.seed,
        ),
        tree=replace(CONFIG.tree, target_trajectories=args.target_trajectories),
        output_path=args.output,
    )

    sampler = IBTreeSampler(config)
    result = sampler.sample(problems, solutions, verbose=True)

    for tree_index, (tree, trajectories) in enumerate(
        zip(result.trees, result.trajectories)
    ):
        print(f"\nIBTree {tree_index} structure")
        print(tree.render(show_text=args.show_text))
        print(f"\nIBTree {tree_index} summary")
        print(json.dumps(tree.summary(), indent=2, ensure_ascii=False))
        if trajectories:
            first = trajectories[0]
            print(
                f"\nTree {tree_index}, trajectory 0 "
                f"(leaf={first.leaf_id}, tokens={len(first.token_ids)})"
            )
            print(first.text)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.export_records(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nToken-wise advantages saved to: {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
