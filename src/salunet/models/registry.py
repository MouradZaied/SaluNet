from typing import Callable

MODEL_REGISTRY: dict[str, Callable] = {}


def register_model(name: str):
    def wrapper(fn: Callable):
        MODEL_REGISTRY[name] = fn
        return fn
    return wrapper


def build_model(name: str, **kwargs):
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    return sorted(MODEL_REGISTRY)
