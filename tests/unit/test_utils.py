import pytest

from hf_model_inspector import field, humanize_params, safe_get


@pytest.mark.parametrize(
    "input_val, expected",
    [
        (None, "N/A"),
        (0, "0.00"),
        (1, "1.00"),
        (999, "999.00"),
        (1000, "1.00K"),
        (1_234, "1.23K"),
        (1_234_567, "1.23M"),
        (1_000_000_000, "1.00B"),
        (1_234_567_890_123, "1.23T"),
        # floats should also be handled (function accepts Optional[int] but works with floats)
        (1234.56, "1.23K"),
        # negative numbers: current implementation does not scale negatives (loop condition is n >= 1000)
        (-1234, "-1234.00"),
    ],
)
def test_humanize_params_various(input_val, expected):
    assert humanize_params(input_val) == expected


def test_humanize_params_boundary_rounding():
    # ensure rounding to two decimals
    assert humanize_params(1499) == "1.50K"  # 1499/1000 = 1.499 -> 1.50K
    assert humanize_params(1500) == "1.50K"


def test_safe_get_nested_present():
    cfg = {"a": {"b": {"c": 42}}}
    assert safe_get(cfg, "a", "b", "c") == 42


def test_safe_get_missing_key_returns_default():
    cfg = {"a": {"b": 1}}
    assert safe_get(cfg, "a", "x", default="missing") == "missing"
    assert safe_get(cfg, "nope", default=None) is None


def test_safe_get_non_dict_intermediate_returns_default():
    cfg = {"a": 123}
    # 'a' is not a dict so attempting to access deeper keys should yield default
    assert safe_get(cfg, "a", "b", default="bad") == "bad"


def test_field_first_non_none_selected():
    cfg = {"x": None, "y": 0, "z": "present"}
    # should skip x (None) and return y (0) even though it's falsy
    assert field(cfg, "x", "y", "z") == 0


def test_field_all_none_or_missing_returns_default():
    cfg = {"a": None}
    assert field(cfg, "a", "b", default="fallback") == "fallback"


def test_field_single_name_present():
    cfg = {"only": "value"}
    assert field(cfg, "only") == "value"
    assert field(cfg, "only") == "value"
