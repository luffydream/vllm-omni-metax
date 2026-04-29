# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
from __future__ import annotations

import os
from typing import Optional

import torch
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm_metax.platform import MacaPlatform

from vllm_omni.diffusion.attention.backends.registry import (
    DiffusionAttentionBackendEnum,
)
from vllm_omni.platforms.interface import OmniPlatform, OmniPlatformEnum

logger = init_logger(__name__)


class MetaxOmniPlatform(OmniPlatform, MacaPlatform):
    """Omni adapter built on top of the existing `vllm-metax` platform.

    This class deliberately *reuses* the concrete MetaX platform from
    `vllm-metax` instead of redefining a new hardware backend.
    """

    # Omni currently has no dedicated METAX enum. CUDA is the closest semantic:
    # a GPU-like backend using torch.cuda style APIs.
    _omni_enum = OmniPlatformEnum.CUDA

    # Preserve the underlying vllm-metax platform identity while making omni's
    # stage device plumbing also mirror MACA_VISIBLE_DEVICES.
    device_control_env_var = getattr(MacaPlatform, "device_control_env_var", "CUDA_VISIBLE_DEVICES")

    @classmethod
    def get_omni_ar_worker_cls(cls) -> str:
        return "vllm_omni.worker.gpu_ar_worker.GPUARWorker"

    @classmethod
    def get_omni_generation_worker_cls(cls) -> str:
        return "vllm_omni.worker.gpu_generation_worker.GPUGenerationWorker"

    @classmethod
    def get_default_stage_config_path(cls) -> str:
        return "vllm_omni/model_executor/stage_configs"

    @classmethod
    def get_diffusion_attn_backend_cls(
        cls,
        selected_backend: str | None,
        head_size: int,
    ) -> str:
        from vllm_omni.diffusion.envs import PACKAGES_CHECKER

        # vllm-omni's built-in CUDA policy uses CUDA compute capability and
        # flash-attn package presence. For MetaX we keep the same conservative
        # policy instead of forcing FLASH_ATTN.
        compute_capability = cls.get_device_capability()
        compute_supported = False
        if compute_capability is not None:
            major, minor = compute_capability
            capability = major * 10 + minor
            compute_supported = 80 <= capability < 100

        packages_info = PACKAGES_CHECKER.get_packages_info()
        packages_available = packages_info.get("has_flash_attn", False)
        flash_attn_supported = compute_supported and packages_available

        if selected_backend is not None:
            backend_upper = selected_backend.upper()
            if backend_upper == "FLASH_ATTN" and not flash_attn_supported:
                if not compute_supported:
                    logger.warning(
                        "Flash Attention requires GPU-like compute capability >= 8.0 "
                        "and < 10.0. Falling back to TORCH_SDPA backend."
                    )
                elif not packages_available:
                    logger.warning(
                        "Flash Attention packages not available. Falling back to TORCH_SDPA backend."
                    )
                logger.info("Defaulting to diffusion attention backend SDPA")
                return DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()
            backend = DiffusionAttentionBackendEnum[backend_upper]
            logger.info("Using diffusion attention backend '%s'", backend_upper)
            return backend.get_path()

        if flash_attn_supported:
            logger.info("Defaulting to diffusion attention backend FLASH_ATTN")
            return DiffusionAttentionBackendEnum.FLASH_ATTN.get_path()

        logger.info("Defaulting to diffusion attention backend SDPA")
        return DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()

    @classmethod
    def supports_torch_inductor(cls) -> bool:
        return True

    @classmethod
    def get_torch_device(cls, local_rank: int | None = None) -> torch.device:
        if local_rank is None:
            return torch.device("cuda")
        return torch.device("cuda", local_rank)

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability | None:
        # Reuse concrete implementation from vllm-metax.
        return MacaPlatform.get_device_capability(device_id)

    @classmethod
    def get_device_count(cls) -> int:
        return MacaPlatform.device_count()

    @classmethod
    def get_device_version(cls) -> str | None:
        # No dedicated version string is exposed by vllm-metax platform.
        return getattr(torch.version, "cuda", None)

    @classmethod
    def synchronize(cls) -> None:
        torch.cuda.synchronize()

    @classmethod
    def get_free_memory(cls, device: torch.device | None = None) -> int:
        free, _ = torch.cuda.mem_get_info(device)
        return int(free)

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return MacaPlatform.get_device_name(device_id)

    @classmethod
    def supports_cpu_offload(cls) -> bool:
        return True

    @classmethod
    def set_device_control_env_var(cls, devices: str | int | None) -> None:
        value = "" if devices is None else str(devices)

        # Keep the owner backend's control variable in sync.
        os.environ[cls.device_control_env_var] = value
        # Mirror the MetaX-specific runtime variable as well. This replaces the
        # source edit previously added in `stage_utils.py`.
        os.environ["MACA_VISIBLE_DEVICES"] = value

    @classmethod
    def unset_device_control_env_var(cls) -> None:
        os.environ.pop(cls.device_control_env_var, None)
        os.environ.pop("MACA_VISIBLE_DEVICES", None)

    @classmethod
    def is_cuda(cls) -> bool:
        # Treat MetaX as omni's GPU/CUDA-like backend.
        return True

    @classmethod
    def get_profiler_cls(cls) -> str:
        return "vllm_omni.profiler.omni_torch_profiler.OmniTorchProfilerWrapper"

