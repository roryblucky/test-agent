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
tool calls, retries, usage, and other details from one model run. Planner,
Specialist, synthesis, tool, and other internal Agent interactions must not be
added to shared Conversation state or inherited by a later request.

Storing PydanticAI messages directly was rejected because LangGraph does not
provide them with `add_messages` merge semantics or built-in strict
deserialization allowlisting. Doing so would require a custom stable-ID reducer
and serializer policy while making it easier to leak actor-local execution
history into the user Conversation. If the application later removes its
LangChain message dependency, the replacement should be a minimal
application-owned Conversation Message contract with an equivalent deterministic
reducer, not raw PydanticAI run history.
