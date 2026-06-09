"""
piai/utils/headers.py — Auth and custom header construction shared across providers.
"""
from __future__ import annotations

import os
from typing import Optional


def get_api_key(
    env_var: str,
    explicit_key: Optional[str] = None,
) -> str:
    """
    Resolve API key from:
      1. explicit_key (from StreamOptions.api_key)
      2. environment variable

    Raises ValueError if neither is available.
    """
    if explicit_key:
        return explicit_key
    key = os.environ.get(env_var)
    if not key:
        raise ValueError(
            f"API key not found. Set {env_var} environment variable "
            f"or pass api_key in StreamOptions."
        )
    return key


def build_auth_headers(api_key: str, scheme: str = "Bearer") -> dict[str, str]:
    """Build an Authorization header dict."""
    return {"Authorization": f"{scheme} {api_key}"}


def merge_headers(*header_dicts: Optional[dict[str, str]]) -> dict[str, str]:
    """
    Merge multiple header dicts left-to-right. Later dicts override earlier ones.
    None values are skipped.
    """
    merged: dict[str, str] = {}
    for headers in header_dicts:
        if headers:
            merged.update(headers)
    return merged
