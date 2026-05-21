<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM-Omni MetaX" src="https://raw.githubusercontent.com/MetaX-MACA/vllm-omni-metax/main/docs/assets/logos/vllm-metax-logo.webp" width="55%">
  </picture>
</p>

<h3 align="center">
vLLM Omni MetaX Plugin
</h3>

<div align="center">

[![Docs](https://img.shields.io/badge/Docs-Read%20the%20Docs-8A2BE2?style=flat&logo=readthedocs&logoColor=white)](https://vllm-omni-metax.readthedocs.io/en/latest/)

</div>

<p align="center">
| <a href="https://www.metax-tech.com/en/"><b>About MetaX</b></a> | <a href="https://vllm-omni-metax.readthedocs.io/en/latest/"><b>Documentation</b></a> | <a href="https://slack.vllm.ai"><b>#sig-maca</b></a> |
</p>

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
- Python 3.10 -- 3.12
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

See [docs](https://vllm-omni-metax.readthedocs.io/en/latest/)
for the full setup flow and runtime patch notes.
