"""Closed structural-limit vocabulary shared by Agent Graph boundaries."""

from typing import Final, Literal, get_args

TASK_LIMIT: Final = "task_limit"
COORDINATION_LIMIT: Final = "coordination_limit"
COORDINATION_INVALID: Final = "coordination_invalid"

StructuralReason = Literal[
    "task_limit",
    "coordination_limit",
    "coordination_invalid",
]
CoordinationStopReason = Literal[
    "task_limit",
    "coordination_limit",
    "coordination_invalid",
]

COORDINATION_STOP_REASONS: Final = frozenset(get_args(CoordinationStopReason))
STRUCTURAL_REASON_ORDER: Final = {
    reason: index for index, reason in enumerate(get_args(StructuralReason))
}
