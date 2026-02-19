from .base import BaseStrategy
from .entropy import EntropyStrategy

_ALL_STRATEGIES = [
    EntropyStrategy,
]

_REGISTRY = {cls().name: cls for cls in _ALL_STRATEGIES}


def get_strategy(name: str) -> BaseStrategy:
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown AL strategy '{name}'. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]()
