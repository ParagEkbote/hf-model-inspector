from typing import Any, Optional

def humanize_params(n: Optional[int]) -> str:
    """Convert a large integer param count into a human-readable format (e.g., 1.23B)."""
    if n is None:
        return "N/A"
    units = ["", "K", "M", "B", "T"]
    k = 0
    while n >= 1000 and k < len(units) - 1:
        n /= 1000
        k += 1
    return f"{n:.2f}{units[k]}"

def safe_get(d: dict, *keys, default=None) -> Any:
    """Safely access nested dict keys: safe_get(cfg, 'a', 'b', 'c')."""
    for key in keys:
        if not isinstance(d, dict) or key not in d:
            return default
        d = d[key]
    return d

def field(cfg: dict, *names: str, default=None) -> Any:
    """Return the first existing non-None field from given names."""
    for name in names:
        if name in cfg and cfg[name] is not None:
            return cfg[name]
    return default
