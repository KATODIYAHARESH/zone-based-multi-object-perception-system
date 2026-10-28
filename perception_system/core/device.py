"""Compute-device selection with graceful CPU fallback."""
from __future__ import annotations

import logging
from typing import List

log = logging.getLogger(__name__)


def select_device(preference: str = "auto") -> str:
    """Return a torch device string, falling back to CPU when the request can't be met."""
    pref = (preference or "auto").strip().lower()
    try:
        import torch
    except ImportError:
        log.warning("PyTorch not installed - running on CPU")
        return "cpu"
    cuda_ok = torch.cuda.is_available()
    mps_ok = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if pref == "auto":
        return "cuda:0" if cuda_ok else ("mps" if mps_ok else "cpu")
    if pref.startswith("cuda"):
        if not cuda_ok:
            log.warning("CUDA requested but unavailable - falling back to CPU")
            return "cpu"
        idx = int(pref.split(":", 1)[1]) if ":" in pref else 0
        if idx >= torch.cuda.device_count():
            log.warning("cuda:%d not present - using cuda:0", idx)
            idx = 0
        return f"cuda:{idx}"
    if pref == "mps":
        if mps_ok:
            return "mps"
        log.warning("MPS requested but unavailable - falling back to CPU")
    return "cpu"


def describe_device(device: str) -> str:
    """Human-readable device label for the HUD."""
    if device.startswith("cuda"):
        try:
            import torch
            return f"{device.upper()} {torch.cuda.get_device_name(int(device.split(':')[1]))}"
        except Exception:  # pragma: no cover - informational only
            return device.upper()
    return device.upper()


def onnx_providers(device: str) -> List[str]:
    """ONNX Runtime execution providers matching the selected device."""
    try:
        import onnxruntime as ort
        available = set(ort.get_available_providers())
    except ImportError:
        return ["CPUExecutionProvider"]
    providers: List[str] = []
    if device.startswith("cuda") and "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers
