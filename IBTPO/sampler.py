"""Standalone vLLM implementation of IBTree structured sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from transformers import AutoTokenizer

from .config import DemoConfig
from .rewards import MathRewardVerifier
from .tree import IBTree, Trajectory


@dataclass(frozen=True)
class SamplingResult:
    trees: List[IBTree]
    generated_per_tree: List[int]
    sampling_rounds: int

    @property
    def generated_trajectories(self) -> int:
        return sum(self.generated_per_tree)

    @property
    def trajectories(self) -> List[List[Trajectory]]:
        return [tree.trajectories() for tree in self.trees]

    @property
    def tree(self) -> IBTree:
        if len(self.trees) != 1:
            raise ValueError("tree is only available for a single-prompt result")
        return self.trees[0]

    def trajectory(self, tree_index: int = 0, trajectory_index: int = 0) -> Trajectory:
        return self.trajectories[tree_index][trajectory_index]

    def export_records(self) -> List[Dict[str, Any]]:
        """Return the minimal user-facing representation of sampled trajectories."""
        return [
            {
                "prompt": tree.problem,
                "prompt_tokens": tree.prompt_token_ids,
                "trajectories": [
                    {
                        "trajectory": trajectory.text,
                        "trajectory_tokens": trajectory.token_ids,
                        "token_wise_trajectory_advantages": trajectory.token_advantages,
                    }
                    for trajectory in trajectories
                ],
            }
            for tree, trajectories in zip(self.trees, self.trajectories)
        ]


@dataclass(frozen=True)
class _PendingRequest:
    tree_index: int
    branch_id: int
    prompt_token_ids: List[int]
    sample_count: int
    max_tokens: int


class IBTreeSampler:
    """Build multiple IBTrees with synchronized batched vLLM rounds."""

    def __init__(self, config: DemoConfig, *, reward_fn: Optional[Any] = None) -> None:
        self.config = config
        self.reward_fn = reward_fn or MathRewardVerifier()
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.engine.model,
            trust_remote_code=config.engine.trust_remote_code,
        )
        self.llm = self._create_engine()

    def _create_engine(self) -> Any:
        from vllm import LLM

        engine = self.config.engine
        return LLM(
            model=engine.model,
            tensor_parallel_size=engine.tensor_parallel_size,
            gpu_memory_utilization=engine.gpu_memory_utilization,
            dtype=engine.dtype,
            max_model_len=engine.max_model_len,
            trust_remote_code=engine.trust_remote_code,
            enable_prefix_caching=engine.enable_prefix_caching,
            disable_log_stats=True,
        )

    def _format_prompt(self, problem: str) -> List[int]:
        """Apply Qwen's ChatML format without depending on ms-swift."""
        system = self.config.system_prompt
        text = ""
        if system:
            text += f"<|im_start|>system\n{system}<|im_end|>\n"
        text += f"<|im_start|>user\n{problem}<|im_end|>\n"
        text += "<|im_start|>assistant\n<think>\n"
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _stop_token_ids(self) -> List[int]:
        candidates = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
        ]
        return sorted({token_id for token_id in candidates if token_id is not None and token_id >= 0})

    @staticmethod
    def _selected_logprobs(output: Any) -> List[float]:
        if output.logprobs is None:
            raise RuntimeError("vLLM did not return token logprobs")
        values: List[float] = []
        for token_id, candidates in zip(output.token_ids, output.logprobs):
            selected = candidates.get(token_id)
            if selected is None:
                raise RuntimeError(f"missing selected-token logprob for token {token_id}")
            values.append(float(selected.logprob))
        return values

    def _generate_batch(
        self,
        pending_requests: Sequence[_PendingRequest],
        round_id: int,
    ) -> List[List[Any]]:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        generation = self.config.generation
        prompts = [
            TokensPrompt(prompt_token_ids=request.prompt_token_ids)
            for request in pending_requests
        ]
        params = [
            SamplingParams(
                n=request.sample_count,
                temperature=generation.temperature,
                top_p=generation.top_p,
                top_k=generation.top_k,
                max_tokens=request.max_tokens,
                seed=generation.seed + round_id * 100_000 + request.tree_index,
                logprobs=1,
                stop_token_ids=self._stop_token_ids(),
            )
            for request in pending_requests
        ]
        request_outputs = self.llm.generate(
            prompts,
            sampling_params=params,
            use_tqdm=False,
        )
        if len(request_outputs) != len(pending_requests):
            raise RuntimeError(
                f"expected {len(pending_requests)} vLLM outputs, got {len(request_outputs)}"
            )
        return [list(request_output.outputs) for request_output in request_outputs]

    @staticmethod
    def _validate_inputs(
        problems: Sequence[str],
        solutions: Optional[Sequence[Optional[str]]],
    ) -> tuple[List[str], List[Optional[str]]]:
        if isinstance(problems, str):
            raise TypeError("problems must be a list or another non-string sequence")
        problem_list = list(problems)
        if not problem_list or not all(isinstance(problem, str) for problem in problem_list):
            raise ValueError("problems must contain at least one string")
        if solutions is None:
            return problem_list, [None] * len(problem_list)
        if isinstance(solutions, str):
            raise TypeError("solutions must be a sequence aligned with problems")
        solution_list = list(solutions)
        if len(solution_list) != len(problem_list):
            raise ValueError("problems and solutions must have equal lengths")
        return problem_list, solution_list

    def sample(
        self,
        problems: Sequence[str],
        solutions: Optional[Sequence[Optional[str]]] = None,
        *,
        verbose: bool = True,
    ) -> SamplingResult:
        problem_list, solution_list = self._validate_inputs(problems, solutions)
        tree_config = self.config.tree
        prompt_token_ids = [self._format_prompt(problem) for problem in problem_list]
        trees = [
            IBTree(
                problem=problem,
                solution=solution,
                prompt_token_ids=prompt_ids,
                tokenizer=self.tokenizer,
                tree_config=tree_config,
                ib_config=self.config.ib,
            )
            for problem, solution, prompt_ids in zip(
                problem_list, solution_list, prompt_token_ids
            )
        ]
        prefix_ids = self.tokenizer.encode(
            tree_config.response_prefix, add_special_tokens=False
        )
        generated_per_tree = [0] * len(trees)
        stopped = [False] * len(trees)
        round_id = 0

        while any(
            not stopped[index]
            and generated_per_tree[index] < tree_config.target_trajectories
            for index in range(len(trees))
        ):
            pending_requests: List[_PendingRequest] = []
            for tree_index, tree in enumerate(trees):
                if stopped[tree_index] or generated_per_tree[tree_index] >= tree_config.target_trajectories:
                    continue
                branch_id = (
                    0
                    if round_id < tree_config.root_branching_rounds
                    else tree.select_branch_node()
                )
                if branch_id is None:
                    stopped[tree_index] = True
                    if verbose:
                        print(f"[tree {tree_index}] no valid branch node remains; stopped early.")
                    continue

                scheduled = (
                    tree_config.branching_schedule[round_id]
                    if round_id < len(tree_config.branching_schedule)
                    else 1
                )
                sample_count = min(
                    scheduled,
                    tree_config.target_trajectories - generated_per_tree[tree_index],
                )
                used_tokens = tree.response_token_count(branch_id)
                token_budget = min(
                    self.config.generation.max_tokens,
                    tree_config.max_response_tokens - used_tokens - len(prefix_ids),
                )
                if sample_count <= 0 or token_budget <= 0:
                    stopped[tree_index] = True
                    continue

                pending_requests.append(
                    _PendingRequest(
                        tree_index=tree_index,
                        branch_id=branch_id,
                        prompt_token_ids=(
                            prompt_token_ids[tree_index]
                            + tree.response_token_ids(branch_id)
                            + prefix_ids
                        ),
                        sample_count=sample_count,
                        max_tokens=token_budget,
                    )
                )
                if verbose:
                    print(
                        f"[round {round_id}] tree={tree_index} branch={branch_id} "
                        f"token_depth={used_tokens} samples={sample_count}"
                    )

            if not pending_requests:
                break
            batched_outputs = self._generate_batch(pending_requests, round_id)
            made_progress = False
            terminal_token_ids = set(self._stop_token_ids())
            for request, outputs in zip(pending_requests, batched_outputs):
                tree = trees[request.tree_index]
                successful = 0
                for output in outputs:
                    generated_ids = list(output.token_ids)
                    generated_logprobs = self._selected_logprobs(output)
                    while generated_ids and generated_ids[-1] in terminal_token_ids:
                        generated_ids.pop()
                        generated_logprobs.pop()
                    output_ids = prefix_ids + generated_ids
                    output_logprobs = [0.0] * len(prefix_ids) + generated_logprobs
                    if tree.add_generated_response(
                        request.branch_id, output_ids, output_logprobs
                    ) is not None:
                        successful += 1
                if successful == 0:
                    stopped[request.tree_index] = True
                    continue
                generated_per_tree[request.tree_index] += successful
                tree.update_metrics(self.reward_fn)
                made_progress = True

            if not made_progress:
                break
            round_id += 1

        for tree, generated in zip(trees, generated_per_tree):
            if generated:
                tree.update_metrics(self.reward_fn)
        return SamplingResult(trees, generated_per_tree, round_id)
