## About

vLLM-Omni MetaX is an adapter plugin that enables **vLLM-Omni** to run on
**MetaX GPUs** by leveraging the existing **vllm-metax backend**.

Unlike `vllm-metax`, which provides the low-level hardware platform
integration, this plugin focuses on bridging the **vLLM-Omni execution stack**
(multi-stage, multimodal pipelines) to the MetaX runtime.

## Responsibilities

| Component | Responsibility |
|----------|---------------|
| vllm-metax | Provides MetaX platform, kernels, and runtime |
| vllm-omni | Handles multi-stage multimodal inference |
| vllm-omni-metax | Bridges Omni to MetaX backend |

## Prerequisites

- MetaX GPU (C-series)
- Linux
- Python >= 3.12
- vLLM (version aligned with vllm-metax)
- vLLM-Omni
- vllm-metax

## Installation

Install vllm-omni-metax as a Python package:

```bash
pip install -e .
```

## Compatibility

The current documentation and adaptation target the `0.20.0` stack:

- `vllm-omni-metax 0.20.0`
- `vllm-omni 0.20.0`
- `vllm-metax 0.20.0`

See [docs/getting_started/installation.md](docs/getting_started/installation.md)
for the full setup flow and runtime patch notes.
