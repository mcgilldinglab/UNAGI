from importlib import import_module

__all__ = ["UNAGI", "preflight_validate", "run_workflow_from_config", "plotting", "perturbations", "monitoring"]


def __getattr__(name):
    if name == "UNAGI":
        return import_module(".UNAGI_tool", __name__).UNAGI
    if name in {"preflight_validate", "run_workflow_from_config"}:
        mod = import_module(".workflow", __name__)
        return getattr(mod, name)
    if name == "plotting":
        return import_module(".plotting", __name__)
    if name == "perturbations":
        return import_module(".perturbations", __name__)
    if name == "monitoring":
        return import_module(".monitoring", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
