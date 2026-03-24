from __future__ import annotations

DEFAULT_DIRECTION_MODE = "two_sided"

_ALIASES = {
    "two_sided": "two_sided",
    "two-sided": "two_sided",
    "two_side": "two_sided",
    "twosided": "two_sided",
    "two_directions": "two_sided",
    "bidirectional": "two_sided",
    "both": "two_sided",
    "one_sided": "one_sided",
    "one-sided": "one_sided",
    "one_side": "one_sided",
    "onesided": "one_sided",
    "single_direction": "one_sided",
    "single-sided": "one_sided",
    "single_sided": "one_sided",
}


def normalize_direction_mode(direction_mode: str | None) -> str:
    if direction_mode is None:
        return DEFAULT_DIRECTION_MODE
    normalized = str(direction_mode).strip().lower()
    if normalized not in _ALIASES:
        raise ValueError(
            "direction_mode must be one of: one_sided, two_sided"
        )
    return _ALIASES[normalized]


def direction_values(bound: float, direction_mode: str | None = None) -> list[float]:
    mode = normalize_direction_mode(direction_mode)
    if mode == "two_sided":
        return [bound, 1 / bound]
    return [bound]


def direction_count(direction_mode: str | None = None) -> int:
    return len(direction_values(2.0, direction_mode))


def infer_direction_mode(
    payload: dict,
    bound: float,
    requested_mode: str | None = None,
) -> str:
    if requested_mode is not None:
        return normalize_direction_mode(requested_mode)
    if str(1 / bound) in payload:
        return "two_sided"
    if str(bound) in payload:
        return "one_sided"
    raise KeyError(
        f"Perturbation result for bound={bound} not found. Available keys: {list(payload.keys())}"
    )


def select_direction_payloads(
    payload: dict,
    bound: float,
    direction_mode: str | None = None,
) -> tuple[str, dict, dict | None]:
    mode = infer_direction_mode(payload, bound, direction_mode)
    forward_key = str(bound)
    if forward_key not in payload:
        raise KeyError(
            f"Perturbation result for bound={bound} not found. Available keys: {list(payload.keys())}"
        )
    reverse_payload = None
    if mode == "two_sided":
        reverse_key = str(1 / bound)
        if reverse_key not in payload:
            raise KeyError(
                f"Two-sided perturbation requested for bound={bound}, but reverse key "
                f"{reverse_key} is missing. Available keys: {list(payload.keys())}"
            )
        reverse_payload = payload[reverse_key]
    return mode, payload[forward_key], reverse_payload


def store_directional_result(
    payload: dict,
    bound: float,
    result: dict,
    direction_mode: str | None = None,
) -> None:
    mode = normalize_direction_mode(direction_mode)
    payload[str(bound)] = result
    if mode == "two_sided":
        payload[str(1 / bound)] = result
