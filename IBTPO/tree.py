"""Efficient, dependency-light tree structure for IBTPO sampling."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .config import IBConfig, TreeConfig
from .parsing import split_reasoning_steps

RewardFunction = Callable[[str, Optional[str]], float]


@dataclass(slots=True)
class IBNode:
    id: int
    depth: int
    parent_id: Optional[int]
    text: str = ""
    token_ids: List[int] = field(default_factory=list)
    token_logprobs: List[float] = field(default_factory=list)
    mean_logprob: Optional[float] = None
    children_ids: List[int] = field(default_factory=list)
    p_star: Optional[float] = None
    covariance: Optional[float] = None
    ib_score: Optional[float] = None
    ib_component: Optional[float] = None
    reward_advantage: Optional[float] = None
    ib_advantage: Optional[float] = None
    advantage: Optional[float] = None

    @property
    def is_leaf(self) -> bool:
        return not self.children_ids

    @property
    def is_branching(self) -> bool:
        return len(self.children_ids) > 1

    @property
    def probability(self) -> Optional[float]:
        return math.exp(self.mean_logprob) if self.mean_logprob is not None else None


@dataclass(frozen=True)
class Trajectory:
    leaf_id: int
    text: str
    token_ids: List[int]
    token_advantages: List[float]


class IBTree:
    """A compact inference tree with O(N) value backpropagation."""

    def __init__(
        self,
        *,
        problem: str,
        solution: Optional[str],
        prompt_token_ids: Sequence[int],
        tokenizer: Any,
        tree_config: TreeConfig,
        ib_config: IBConfig,
    ) -> None:
        if tree_config.max_depth <= 0:
            raise ValueError("max_depth must be positive")
        if ib_config.alpha == 1.0:
            raise ValueError("IB alpha must differ from 1")
        if ib_config.beta <= 0.0:
            raise ValueError("IB beta must be positive")
        self.problem = problem
        self.solution = solution
        self.prompt_token_ids = list(prompt_token_ids)
        self.tokenizer = tokenizer
        self.tree_config = tree_config
        self.ib_config = ib_config
        self.nodes: Dict[int, IBNode] = {
            0: IBNode(
                id=0,
                depth=0,
                parent_id=None,
                text=problem,
                token_ids=list(prompt_token_ids),
            )
        }
        self._next_id = 1
        self._end_think_ids = tokenizer.encode(
            tree_config.end_think_marker, add_special_tokens=False
        )
        self._separator_ids = [
            tokenizer.encode(separator, add_special_tokens=False)
            for separator in tree_config.step_separators
        ]

    @property
    def root(self) -> IBNode:
        return self.nodes[0]

    def add_node(
        self,
        *,
        parent_id: int,
        text: str,
        token_ids: Sequence[int],
        token_logprobs: Sequence[float],
        mean_logprob: float,
    ) -> IBNode:
        if parent_id not in self.nodes:
            raise KeyError(f"unknown parent node: {parent_id}")
        parent = self.nodes[parent_id]
        if parent.depth >= self.tree_config.max_depth:
            raise ValueError("cannot add a node below max_depth")
        node = IBNode(
            id=self._next_id,
            depth=parent.depth + 1,
            parent_id=parent_id,
            text=text,
            token_ids=list(token_ids),
            token_logprobs=list(token_logprobs),
            mean_logprob=mean_logprob,
        )
        self.nodes[node.id] = node
        parent.children_ids.append(node.id)
        self._next_id += 1
        return node

    def path_nodes(self, node_id: int, *, include_root: bool = True) -> List[IBNode]:
        if node_id not in self.nodes:
            raise KeyError(f"unknown node: {node_id}")
        path: List[IBNode] = []
        current: Optional[IBNode] = self.nodes[node_id]
        while current is not None:
            path.append(current)
            current = self.nodes[current.parent_id] if current.parent_id is not None else None
        path.reverse()
        return path if include_root else path[1:]

    def response_text(self, node_id: int) -> str:
        return "".join(node.text for node in self.path_nodes(node_id, include_root=False))

    def response_token_ids(self, node_id: int) -> List[int]:
        return [
            token
            for node in self.path_nodes(node_id, include_root=False)
            for token in node.token_ids
        ]

    def response_token_count(self, node_id: int) -> int:
        return sum(len(node.token_ids) for node in self.path_nodes(node_id, include_root=False))

    def leaf_ids(self, node_id: int = 0) -> List[int]:
        if node_id not in self.nodes:
            raise KeyError(f"unknown node: {node_id}")
        leaves: List[int] = []
        stack = [node_id]
        while stack:
            current_id = stack.pop()
            current = self.nodes[current_id]
            if current.is_leaf:
                leaves.append(current_id)
            else:
                stack.extend(reversed(current.children_ids))
        return leaves

    def add_generated_response(
        self,
        parent_id: int,
        token_ids: Sequence[int],
        logprobs: Sequence[float],
    ) -> Optional[int]:
        """Parse and append one generated continuation; return its final node."""
        parent = self.nodes[parent_id]
        remaining_depth = self.tree_config.max_depth - parent.depth
        remaining_tokens = (
            self.tree_config.max_response_tokens - self.response_token_count(parent_id)
        )
        if remaining_depth <= 0 or remaining_tokens <= 0:
            return None

        limited_ids = list(token_ids[:remaining_tokens])
        limited_logprobs = list(logprobs[:remaining_tokens])
        steps = split_reasoning_steps(
            limited_ids,
            limited_logprobs,
            max_steps=remaining_depth,
            end_think_ids=self._end_think_ids,
            separator_ids=self._separator_ids,
            min_step_tokens=self.tree_config.min_step_tokens,
            max_step_tokens=self.tree_config.max_step_tokens,
        )
        current_id = parent_id
        for step in steps:
            node = self.add_node(
                parent_id=current_id,
                text=self.tokenizer.decode(step.token_ids, skip_special_tokens=False),
                token_ids=step.token_ids,
                token_logprobs=step.logprobs,
                mean_logprob=step.mean_logprob,
            )
            current_id = node.id
        return current_id if current_id != parent_id else None

    def update_metrics(self, reward_fn: RewardFunction) -> None:
        """Update p*, IB-Score and advantages in linear time."""
        reward_floor = self.ib_config.reward_floor
        leaf_sum: Dict[int, float] = {}
        leaf_count: Dict[int, int] = {}

        for node in reversed(list(self.nodes.values())):
            if node.is_leaf:
                if node.id == 0:
                    reward = 0.0
                elif node.p_star is None:
                    reward = float(reward_fn(self.response_text(node.id), self.solution))
                else:
                    reward = node.p_star
                node.p_star = max(reward_floor, reward)
                leaf_sum[node.id] = node.p_star
                leaf_count[node.id] = 1
            else:
                total = sum(leaf_sum[child_id] for child_id in node.children_ids)
                count = sum(leaf_count[child_id] for child_id in node.children_ids)
                node.p_star = total / count
                leaf_sum[node.id] = total
                leaf_count[node.id] = count

        self._update_ib_scores()
        self._update_advantages()

    def _edge_terms(self, parent: IBNode, child: IBNode) -> tuple[float, float]:
        assert parent.p_star is not None and child.p_star is not None
        assert child.probability is not None
        alpha = self.ib_config.alpha
        beta = self.ib_config.beta
        eta_reward = (child.p_star / (parent.p_star + 1e-6)) ** (alpha - 1.0)
        eta_reward -= 1.0 + 1.0 / beta
        eta_policy = child.probability ** (alpha - 1.0)
        return eta_reward, eta_policy

    def _update_ib_scores(self) -> None:
        alpha = self.ib_config.alpha
        beta = self.ib_config.beta
        for node in self.nodes.values():
            if node.parent_id is None:
                node.ib_component = None
            else:
                eta_reward, eta_policy = self._edge_terms(
                    self.nodes[node.parent_id], node
                )
                node.ib_component = eta_reward * eta_policy

            if node.is_leaf:
                node.ib_score = None
                node.covariance = None
                continue
            terms = [self._edge_terms(node, self.nodes[child]) for child in node.children_ids]
            products = [left * right for left, right in terms]
            node.ib_score = (1.0 + beta * statistics.fmean(products)) / (alpha - 1.0)
            if len(terms) < 2:
                node.covariance = 0.0
            else:
                left_mean = statistics.fmean(left for left, _ in terms)
                right_mean = statistics.fmean(right for _, right in terms)
                node.covariance = sum(
                    (left - left_mean) * (right - right_mean)
                    for left, right in terms
                ) / (len(terms) - 1)

    def _update_advantages(self) -> None:
        leaf_values = [self.nodes[node_id].p_star for node_id in self.leaf_ids()]
        valid_values = [value for value in leaf_values if value is not None]
        mean = statistics.fmean(valid_values) if valid_values else 0.0
        std = statistics.stdev(valid_values) if len(valid_values) > 1 else 0.0
        eps = self.ib_config.normalization_eps

        for node in self.nodes.values():
            if node.id == 0:
                node.reward_advantage = None
                node.ib_advantage = None
                node.advantage = None
                continue
            node.reward_advantage = (
                (node.p_star - mean) / (std + eps)
                if node.p_star is not None and std > 0.0
                else 0.0
            )
            parent = self.nodes[node.parent_id]  # type: ignore[index]
            node.ib_advantage = node.ib_component if parent.is_branching else 0.0
            node.advantage = (
                node.reward_advantage
                + self.ib_config.advantage_weight * (node.ib_advantage or 0.0)
            )

    def select_branch_node(self) -> Optional[int]:
        """Select the highest-IB valid branch point, or stop if none exists."""
        if not self.root.children_ids:
            return 0
        candidates: List[IBNode] = []
        for node in self.nodes.values():
            if node.is_leaf or node.depth >= self.tree_config.max_depth:
                continue
            if len(node.children_ids) >= self.tree_config.max_children_per_node:
                continue
            if self.response_token_count(node.id) > self.tree_config.max_branching_token_depth:
                continue
            if node.ib_score is not None and math.isfinite(node.ib_score):
                candidates.append(node)
        if not candidates:
            return None
        return max(candidates, key=lambda node: (node.ib_score, -node.id)).id

    def trajectory(self, leaf_id: int) -> Trajectory:
        leaf = self.nodes.get(leaf_id)
        if leaf is None or not leaf.is_leaf or leaf_id == 0:
            raise ValueError("trajectory endpoint must be a non-root leaf")
        path = self.path_nodes(leaf_id, include_root=False)
        token_ids: List[int] = []
        token_advantages: List[float] = []
        for node in path:
            token_ids.extend(node.token_ids)
            token_advantages.extend([float(node.advantage or 0.0)] * len(node.token_ids))
        return Trajectory(
            leaf_id=leaf_id,
            text="".join(node.text for node in path),
            token_ids=token_ids,
            token_advantages=token_advantages,
        )

    def trajectories(self) -> List[Trajectory]:
        return [self.trajectory(node_id) for node_id in self.leaf_ids() if node_id != 0]

    def summary(self) -> Dict[str, Any]:
        leaves = [self.nodes[node_id] for node_id in self.leaf_ids()]
        response_lengths = [self.response_token_count(node.id) for node in leaves]
        rewards = [node.p_star for node in leaves if node.p_star is not None]
        return {
            "nodes": len(self.nodes),
            "leaves": len(leaves),
            "max_depth": max((node.depth for node in self.nodes.values()), default=0),
            "response_tokens_min": min(response_lengths, default=0),
            "response_tokens_mean": statistics.fmean(response_lengths) if response_lengths else 0.0,
            "response_tokens_max": max(response_lengths, default=0),
            "leaf_p_star_mean": statistics.fmean(rewards) if rewards else None,
        }

    @staticmethod
    def _node_label(node: IBNode, show_text: bool, text_width: int) -> str:
        probability = f"{node.probability:.3f}" if node.probability is not None else "-"
        p_star = f"{node.p_star:.3f}" if node.p_star is not None else "-"
        ib_score = f"{node.ib_score:.3f}" if node.ib_score is not None else "-"
        advantage = f"{node.advantage:.3f}" if node.advantage is not None else "-"
        label = (
            f"[{node.id}] depth={node.depth} tokens={len(node.token_ids)} "
            f"p={probability} p*={p_star} IB={ib_score} A={advantage}"
        )
        if show_text and node.id != 0:
            preview = node.text.replace("\n", "\\n")
            label += f' text="{preview[:text_width]}"'
        return label

    def render(self, *, show_text: bool = False, text_width: int = 72) -> str:
        lines = [self._node_label(self.root, show_text, text_width)]

        def visit(node_id: int, prefix: str) -> None:
            children = self.nodes[node_id].children_ids
            for index, child_id in enumerate(children):
                is_last = index == len(children) - 1
                lines.append(
                    prefix
                    + ("└── " if is_last else "├── ")
                    + self._node_label(self.nodes[child_id], show_text, text_width)
                )
                visit(child_id, prefix + ("    " if is_last else "│   "))

        visit(0, "")
        return "\n".join(lines)
