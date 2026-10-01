"""Supported XAI method identifiers."""

from __future__ import annotations

from typing import Any


SUPPORTED_XAI_METHODS = ["shap", "lime", "integrated_gradients", "occlusion"]


def configured_xai_methods(cfg: dict[str, Any]) -> list[str]:
    """Return primary plus secondary XAI methods, preserving config order."""
    methods: list[str] = []
    for key in ["xai_methods", "secondary_xai_methods"]:
        for method in cfg.get(key, []):
            method_id = str(method)
            if method_id not in SUPPORTED_XAI_METHODS:
                raise ValueError(
                    f"Unsupported XAI method '{method_id}'. "
                    f"Supported methods are: {', '.join(SUPPORTED_XAI_METHODS)}"
                )
            if method_id not in methods:
                methods.append(method_id)
    return methods
