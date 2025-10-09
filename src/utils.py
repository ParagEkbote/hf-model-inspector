# utils.py
from typing import Optional

def humanize_params(n: Optional[int]) -> str:
    """Convert large integer param counts into human-readable format."""
    if n is None:
        return "N/A"
    if n >= 1_000_000_000_000:
        return f"~{n/1_000_000_000_000:.2f}T"
    if n >= 1_000_000_000:
        return f"~{n/1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"~{n/1_000_000:.2f}M"
    return str(n)

def safe_get(d: dict, *keys, default=None):
    """Safe nested dictionary access."""
    for k in keys:
        if d is None:
            return default
        if k in d:
            return d[k]
    return default

def field(cfg, *names):
    """Helper to get nested fields with fallback."""
        for n in names:
            if n in cfg and cfg[n] is not None:
                    return cfg[n]
            return None
