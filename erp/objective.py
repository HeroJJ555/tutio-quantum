from enum import StrEnum

from erp.instance import Instance
from erp.schedule import Schedule, makespan, weighted_tardiness


class Objective(StrEnum):
    MAKESPAN = "makespan"
    TARDINESS = "tardiness"


def score(objective: Objective, instance: Instance, schedule: Schedule) -> int:
    if objective is Objective.TARDINESS:
        return weighted_tardiness(instance, schedule)
    return makespan(schedule)
