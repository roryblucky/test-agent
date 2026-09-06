"""Closed structural-limit vocabulary shared by Agent Graph boundaries."""

from typing import Final, Literal, get_args

TASK_LIMIT: Final = "task_limit"
COORDINATION_LIMIT: Final = "coordination_limit"
COORDINATOR_CONTEXT_LIMIT: Final = "coordinator_context_limit"
COORDINATION_INVALID: Final = "coordination_invalid"
CALCULATION_STATE_LIMIT: Final = "calculation_state_limit"
PREPARED_SYNTHESIS_LIMIT: Final = "prepared_synthesis_limit"

StructuralReason = Literal[
    "task_limit",
    "coordination_limit",
    "coordinator_context_limit",
    "coordination_invalid",
    "calculation_state_limit",
    "prepared_synthesis_limit",
]
CoordinationStopReason = Literal[
    "task_limit",
    "coordination_limit",
    "coordinator_context_limit",
    "coordination_invalid",
    "calculation_state_limit",
]

COORDINATION_STOP_REASONS: Final = frozenset(get_args(CoordinationStopReason))
STRUCTURAL_REASON_ORDER: Final = {
    reason: index for index, reason in enumerate(get_args(StructuralReason))
}
