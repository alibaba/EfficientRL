# IBTPO Sampling Demo

A standalone demo of IBTree structured sampling and token-wise IBTPO advantages. It uses vLLM to sample one or more prompts, prints each inference tree, and saves complete trajectories with token-level advantages.

## Requirements

- Linux with an NVIDIA GPU and a compatible CUDA environment
- Python 3.10 or later
- vLLM 0.11 or later
- Transformers 4.57 or later

## Installation

```bash
conda create -n ibtpo-demo python=3.10 -y
conda activate ibtpo-demo
pip install -r requirements.txt
pip install -e .
```

The default model is `Qwen/Qwen3-8B-Base` and will be downloaded from Hugging Face. A local model path can be supplied with `--model`.

## Run

Run the default MATH-500 example:

```bash
ibtpo-demo
```

Run with a local model:

```bash
ibtpo-demo --model /path/to/Qwen3-8B-Base
```

Sample multiple prompts in the same batched inference rounds:

```bash
ibtpo-demo \
  --prompt "Compute 1+1. Put the answer in \\boxed{}." \
  --solution "\\boxed{2}" \
  --prompt "Compute 2+3. Put the answer in \\boxed{}." \
  --solution "\\boxed{5}"
```

Useful options include `--target-trajectories`, `--temperature`, `--top-p`, `--top-k`, `--max-tokens`, `--tensor-parallel-size`, and `--output`.

## Output

The default output is `outputs/token_advantages.json`. For each prompt it contains only:

- `prompt`
- `prompt_tokens`
- `trajectories`
  - `trajectory`
  - `trajectory_tokens`
  - `token_wise_trajectory_advantages`

The output directory is created automatically and ignored by Git.
