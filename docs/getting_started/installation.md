# Installation

This page follows the same high-level approach as `vLLM-metax`: prepare a valid
MetaX runtime first, then install the Omni-side adapter on top.

## Requirements

- OS: Linux
- Python: 3.10 -- 3.12
- Hardware: MetaX C-series GPU
- SDK: MACA runtime correctly installed
- Base software:
  - `vllm-metax`
  - `vllm-omni`

## Compatibility notes

`vllm-omni-metax` is a thin Python plugin. It does not ship extra kernels by
itself and therefore depends on a healthy `vllm-metax` installation. In the
`0.20.0` release line it also ships a lightweight runtime patch for Omni rotary
embedding import paths that otherwise assume CUDA-only flash-attn bindings.

For this repository revision:

- `vllm-omni-metax`: `0.20.0`
- `vllm-omni`: `0.20.0`
- `vllm_metax`: `0.20.0`

When in doubt, keep the three components aligned to the same release train used
by your MetaX environment or Docker image.

## Prepare the MetaX environment

If you already have a working `vllm-metax` container or host environment, you
can reuse it directly.

When building from source, the common MACA environment variables typically look
like this:

```bash
export MACA_PATH="/opt/maca"
export CUCC_PATH="${MACA_PATH}/tools/cu-bridge"
export CUDA_PATH="${HOME}/cu-bridge/CUDA_DIR"
export CUCC_CMAKE_ENTRY=2

export PATH="${MACA_PATH}/mxgpu_llvm/bin:${MACA_PATH}/bin:${CUCC_PATH}/tools:${CUCC_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/ompi/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH}"
```

For the detailed backend-side build flow, refer to the
[`vLLM-metax` installation document](https://vllm-metax.readthedocs.io/en/latest/getting_started/installation/maca.html).

## Install `vllm-metax`

`vllm-omni-metax` assumes the MetaX backend is already available. Install
`vllm-metax` first using your preferred method:

- MetaX released Docker image for the matching MACA version
- Editable source install for development
- Prebuilt wheel inside an existing MetaX software environment

## Install `vllm-omni`

Install `vllm-omni` in the same Python environment that already contains
`vllm-metax`.

If your team already maintains a known-good `vllm-omni` checkout, use that
exact revision. Otherwise, keep it aligned with the version required by this
package.

## Install `vllm-omni-metax`

Clone this repository and install it into the same environment:

```bash
git clone https://github.com/MetaX-MACA/vllm-omni-metax.git
cd vllm-omni-metax
git checkout the/your/choice/branch
pip install -e .
```

If you use `uv`, the equivalent command is:

```bash
uv pip install -e .
```

Because this package is only an adapter layer, editable mode is the recommended
choice during bring-up and debugging.

