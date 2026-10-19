"""Configuration for the standalone IBTree sampling demo."""

from dataclasses import dataclass, field
from typing import Tuple

DEFAULT_MODEL = "Qwen/Qwen3-8B-Base"
DEFAULT_PROBLEM = (
    "If $f(x) = \\frac{3x-2}{x-2}$, what is the value of "
    "$f(-2) + f(-1) + f(0)$? Express your answer as a common fraction."
)
DEFAULT_SOLUTION = (
    "We have $f(-2)=2$, $f(-1)=\\frac{5}{3}$, and $f(0)=1$. "
    "Therefore the sum is $2+\\frac{5}{3}+1=\\boxed{\\frac{14}{3}}$."
)
DEFAULT_PROMPTS = [DEFAULT_PROBLEM]
DEFAULT_SOLUTIONS = [DEFAULT_SOLUTION]
DEFAULT_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

STEP_SEPARATORS: Tuple[str, ...] = (
    "\n\n",
    ".\n\n",
    ";\n\n",
    ":\n\n",
    "$\n\n",
    "%\n\n",
    ">\n\n",
    ")\n\n",
    "]\n\n",
    "}\n\n",
    "!\n\n",
    " }\n\n",
    ");\n\n",
    "\n\n\n\n",
    "\n\n\n",
    "();\n\n",
    " {\n\n",
    '"\n\n',
    "?\n\n",
    ".\n\n\n\n",
)


@dataclass(frozen=True)
class EngineConfig:
    model: str = DEFAULT_MODEL
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.55
    dtype: str = "bfloat16"
    max_model_len: int = 4096
    trust_remote_code: bool = True
    enable_prefix_caching: bool = True


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 20
    max_tokens: int = 2048
    seed: int = 42


@dataclass(frozen=True)
class TreeConfig:
    target_trajectories: int = 12
    max_depth: int = 8
    max_response_tokens: int = 2048
    min_step_tokens: int = 64
    max_step_tokens: int = 512
    max_branching_token_depth: int = 256
    max_children_per_node: int = 3
    branching_schedule: Tuple[int, ...] = (4, 1, 1, 1, 1, 1, 1, 1, 1)
    root_branching_rounds: int = 1
    response_prefix: str = ""
    step_separators: Tuple[str, ...] = STEP_SEPARATORS
    end_think_marker: str = "</think>"


@dataclass(frozen=True)
class IBConfig:
    alpha: float = 2.0
    beta: float = 5.0
    advantage_weight: float = 0.1
    reward_floor: float = 1e-3
    normalization_eps: float = 1e-4


@dataclass(frozen=True)
class DemoConfig:
    engine: EngineConfig = field(default_factory=EngineConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    tree: TreeConfig = field(default_factory=TreeConfig)
    ib: IBConfig = field(default_factory=IBConfig)
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    output_path: str = "outputs/token_advantages.json"


CONFIG = DemoConfig()
