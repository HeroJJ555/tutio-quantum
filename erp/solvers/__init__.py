from dataclasses import dataclass, field

from erp.schedule import Schedule


@dataclass
class QuboResult:
    bits: list[int] | None  # None if no feasible sample was found
    schedule: Schedule | None
    energy: float | None
    method: str
    stats: dict = field(default_factory=dict)
