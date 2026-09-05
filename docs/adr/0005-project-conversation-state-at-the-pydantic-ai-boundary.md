# Project Conversation state at the PydanticAI actor boundary

Status: accepted

Checkpoint `conversation_messages` uses LangChain `HumanMessage` and
`AIMessage` values with stable request-and-role IDs. This is the LangGraph state
representation, not the product History model: it deliberately uses LangGraph's
native `add_messages` reducer for idempotent merge-by-ID behavior and the
checkpointer's built-in safe serialization support. Only the logical user
message and the final publishable assistant message enter this state.

Before invoking a PydanticAI actor, application code selects complete prior
user/assistant pairs within the context budget and projects them to PydanticAI
`ModelRequest` and `ModelResponse` values. PydanticAI messages remain an actor
boundary type because they can also contain provider metadata, instructions,
tool calls, retries, usage, and other details from one model run. Coordinator,
Specialist, synthesis, tool, and other internal Agent interactions must not be
added to shared Conversation state or inherited by a later request.

In Agent mode, only the Query Understanding actor receives that projected prior
Conversation context. It resolves the current request into a standalone query
and selected Business Intent. Coordinator, Specialist, and Synthesis actors
receive deterministic role-specific projections of that resolved current-Run
state and never receive the prior Conversation pairs. Linear-mode actor inputs
remain governed by the Linear Graph's existing boundaries.

Every new Agent Run first passes through one unique initializer. It preserves
only checkpointed Conversation Messages, clears reducer-backed Run-local state
with LangGraph overwrite semantics, and resets scalar channels with ordinary
values before any parallel branch can run. This prevents a new Request from
inheriting accepted batches, staging, answers, or diagnostics from the
previous Run.

Final publication uses the inverse boundary: deterministic gates build one
canonical response, a final-state node checkpoints that response, its citations,
publication manifest, and final assistant Message with synchronous durability,
and only then may a publication node emit answer frames. The network stream is a
projection of committed state and is not an exactly-once or replay contract.

Storing PydanticAI messages directly was rejected because LangGraph does not
provide them with `add_messages` merge semantics or built-in strict
deserialization allowlisting. Doing so would require a custom stable-ID reducer
and serializer policy while making it easier to leak actor-local execution
history into the user Conversation. If the application later removes its
LangChain message dependency, the replacement should be a minimal
application-owned Conversation Message contract with an equivalent deterministic
reducer, not raw PydanticAI run history.
