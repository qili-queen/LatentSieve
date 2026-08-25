import os
import re
from pathlib import Path

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline


def configure_qi_runtime() -> None:
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.pop("HF_ENDPOINT", None)


def _read_nvidia_kernel_version() -> str | None:
    version_path = Path("/proc/driver/nvidia/version")
    if not version_path.exists():
        return None
    text = version_path.read_text(errors="ignore")
    match = re.search(r"Kernel Module\s+(\d+\.\d+\.\d+)", text)
    return match.group(1) if match else None


def _read_nvidia_userspace_version() -> str | None:
    for candidate in (
        Path("/usr/lib/x86_64-linux-gnu/libcuda.so.1"),
        Path("/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"),
    ):
        if candidate.exists():
            resolved = candidate.resolve().name
            match = re.search(r"(\d+\.\d+\.\d+)", resolved)
            if match:
                return match.group(1)
    return None


def explain_cuda_unavailable(reason: Exception | None = None) -> str:
    kernel_version = _read_nvidia_kernel_version()
    userspace_version = _read_nvidia_userspace_version()

    message_parts = []
    if reason is not None:
        message_parts.append(f"torch CUDA probe failed: {reason}")
    else:
        message_parts.append("torch CUDA probe failed")

    if kernel_version and userspace_version and kernel_version != userspace_version:
        message_parts.append(
            "system NVIDIA driver mismatch: "
            f"kernel module={kernel_version}, userspace libraries={userspace_version}"
        )
    elif kernel_version or userspace_version:
        message_parts.append(
            "NVIDIA versions observed: "
            f"kernel module={kernel_version or 'unknown'}, userspace libraries={userspace_version or 'unknown'}"
        )

    message_parts.append(
        "This is a system-level driver/library problem, not a project code issue. "
        "Fix by aligning the installed NVIDIA userspace packages with the loaded kernel module, "
        "or by rebooting after the driver update so the kernel module matches the installed libraries."
    )
    return " ".join(message_parts)


def resolve_qi_device(gpu_index: int | None) -> tuple[str, torch.dtype]:
    if gpu_index is None:
        return "cpu", torch.float32

    try:
        device_count = torch.cuda.device_count()
    except Exception as exc:
        raise RuntimeError(explain_cuda_unavailable(exc)) from exc

    if device_count <= gpu_index:
        raise RuntimeError(explain_cuda_unavailable())

    return f"cuda:{gpu_index}", torch.float32


def build_cached_pipe(model_name: str, device: str, dtype: torch.dtype):
    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        model_name,
        subfolder="scheduler",
        local_files_only=True,
    )
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        scheduler=scheduler,
        torch_dtype=dtype,
        safety_checker=None,
        local_files_only=True,
    ).to(device)
    return pipe, scheduler
