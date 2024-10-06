from dataclasses import dataclass
from collections.abc import Sequence
from optuna.distributions import CategoricalChoiceType


@dataclass
class IntSuggestions:
    name: str
    low: int
    high: int


@dataclass
class FloatSuggestions:
    name: str
    low: float
    high: float


@dataclass
class CategoricalSuggestions:
    name: str
    choices: Sequence[CategoricalChoiceType]
