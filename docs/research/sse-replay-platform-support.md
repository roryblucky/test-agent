# AI streaming replay support

Research date: 2026-09-04.

## Terminology

This note distinguishes three behaviors that are often all called "resume":

1. **SSE replay**: the server persists emitted events and, on reconnect, sends only events after an SSE `id` / HTTP `Last-Event-ID` cursor.
2. **Equivalent cursor-based stream resumption**: the service persists a background response or durable stream, and the client reconnects with a response/run ID plus a provider-specific sequence cursor. This is functionally event replay, but not necessarily the standard `Last-Event-ID` wire mechanism.
3. **Recovery without replay**: the client fetches the final object, restores conversation state, or submits a new generation request seeded with partial output. Missed deltas are not replayed.

## Summary

| Platform/API | Classification | What reconnects | Important scope |
| --- | --- | --- | --- |
| LangGraph Agent Server | **1 — standard SSE replay** | `Last-Event-ID` resumes after the last received event | A run must be created with `stream_resumable=true`; thread streams also expose `Last-Event-ID` |
| OpenAI Responses API | **2 — exact cursor replay** | `response_id` + event `sequence_number` via `starting_after` | Only background Responses originally created with `stream=true` |
| Gemini Interactions API / Deep Research | **2 — exact cursor replay** | `interaction_id` + event `event_id` via `last_event_id` | Interactions API is beta; Deep Research is preview |
| Vercel Workflow + AI SDK | **2 — durable chunk replay** | `runId` + `startIndex`; or an application-managed resumable stream | Framework/platform feature, not a capability inherited from the model provider |
| Azure OpenAI Responses API | **2 — exact cursor replay** | Response ID + `sequence_number` via `starting_after` | Background Responses originally created with `stream=true` |
| Anthropic Messages API | **3 — new inference continuation** | A new request containing the received partial text | Not the same generation; partial tool/thinking blocks cannot be recovered |
| Anthropic Managed Agents beta | **3 — final-event recovery only** | Reopen stream and list persisted event history | Live preview deltas are explicitly not persisted or replayed |
| Gemini Live API / Vertex Live API | **3 — session/context resumption** | WebSocket session handle restores conversational state | Does not promise replay of missed model-output frames; some states may lose data |
| Vertex `streamGenerateContent` | **No documented replay** | N/A | Public API documents a live chunk stream, not a response/event cursor |
| Amazon Bedrock `ConverseStream` / `InvokeModelWithResponseStream` | **No documented replay** | N/A | Public APIs expose one live AWS event stream and tell clients to retry stream errors; no response ID + event cursor is documented |

## Confirmed replay implementations

### LangGraph Agent Server

LangGraph provides the closest match to standards-based SSE replay. The run-stream endpoint accepts `Last-Event-ID`; if the run was created with `stream_resumable=true`, the server sends events after the last seen ID. The thread-stream endpoint likewise documents `Last-Event-ID`, with `-` meaning replay from the beginning. The Agent Server v0.6.0 changelog explicitly says the semantics were aligned with the SSE specification so the event named by `Last-Event-ID` is excluded and only following events are returned.

Sources: [Join Run Stream](https://docs.langchain.com/langsmith/agent-server-api/thread-runs/join-run-stream), [Join Thread Stream](https://docs.langchain.com/langsmith/agent-server-api/threads/join-thread-stream), [Agent Server changelog](https://langchain-ai.github.io/langgraph/cloud/reference/langgraph_server_changelog/).

This also explains an apparent documentation conflict: ordinary `join_stream` output is described as unbuffered, so output produced before joining is lost. Replay is available only on the resumable path/configuration; it should not be inferred from every LangGraph stream. Source: [LangGraph Streaming API](https://langchain-ai.github.io/langgraphjs/cloud/how-tos/stream_messages/).

### OpenAI Responses API

An OpenAI background Response created with both `background=true` and `stream=true` continues running after the original connection drops. Each event carries `sequence_number`; reconnecting with `GET /responses/{response_id}?stream=true&starting_after=N` streams events after `N`. This is exact event-level resumption using an application cursor, although the cursor is a query parameter rather than the standard HTTP `Last-Event-ID` header. The public guide notes that some SDK convenience methods are still language/version dependent, while the REST endpoint and API parameter are documented.

Sources: [OpenAI background mode](https://developers.openai.com/api/docs/guides/background), [Retrieve a Response API reference](https://developers.openai.com/api/reference/cli/resources/responses/methods/retrieve).

### Gemini Interactions API

Gemini's background Interactions stream associates events with `event_id`. A dropped connection can reconnect to the same interaction using `stream=true&last_event_id=...`; the API resumes from the next chunk. Google's Deep Research example tracks both `interaction_id` and `last_event_id` and reconnects automatically, including after the documented 600-second streaming connection timeout. This is exact provider-specific cursor replay over SSE, not HTTP `Last-Event-ID`.

Sources: [Gemini Deep Research streaming and reconnection](https://ai.google.dev/gemini-api/docs/deep-research), [Google Cloud Interactions API](https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/models/interactions-api).

### Vercel Workflow and AI SDK

Vercel supports resumption at the application/framework layer in two related forms:

- `WorkflowChatTransport` retains a workflow run ID and reconnects to `{api}/{runId}/stream` with `startIndex`; the server returns `x-workflow-stream-tail-index`, enabling the next reconnect to continue from the last chunk.
- AI SDK `useChat` can reconnect to an active stream, but the application must provide stream persistence, an active-stream mapping, Redis-backed `resumable-stream`, and create/resume endpoints. It is therefore a supported construction kit, not automatic replay supplied by an upstream model API.

Sources: [Vercel WorkflowAgent resumable streaming](https://vercel.com/kb/guide/what-is-workflowagent), [AI SDK chatbot resume streams](https://ai-sdk.dev/docs/ai-sdk-ui/chatbot-resume-streams).

### Azure OpenAI Responses API

Microsoft documents the same background-Responses cursor model: record each event's `sequence_number`, then reconnect with `stream=true&starting_after=N`; the service replays events after that sequence number. The original response must have been created with `stream=true`.

Source: [Azure OpenAI Responses API](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/responses).

## Things called "resume" that are not SSE replay

### Anthropic

The Claude Messages API's documented error recovery creates a **new** Messages request using the partial text already received. For Claude 4.5 and earlier, the partial response can seed an assistant message; for Claude 4.6 and later, the client sends a user message containing the partial response and asks Claude to continue. Anthropic explicitly notes that partial tool-use and thinking blocks cannot be recovered. This is another inference that attempts a natural continuation, not replay of the original stream.

Source: [Anthropic streaming — error recovery](https://platform.claude.com/docs/en/build-with-claude/streaming#error-recovery).

Anthropic Managed Agents beta persists authoritative buffered events such as the complete `agent.message`, so a reconnect can list events emitted during disconnection. However, its incremental `event_start` / `event_delta` previews are explicitly best-effort, never persisted, and unavailable for replay after reconnect. Thus it offers final-event/history recovery, not token/delta replay.

Source: [Managed Agents events and streaming](https://platform.claude.com/docs/en/managed-agents/events-and-streaming).

### Google Live API

Gemini/Vertex Live API can resume a WebSocket session using a `SessionResumptionUpdate.new_handle`; transparent mode reports the last consumed **client** message index so the client can resend unacknowledged input. This restores conversation/session context. It is not an event cursor for replaying missed server output, and the reference warns that resuming during non-resumable states such as generation or function calls can lose data.

Sources: [Gemini Live API session management](https://ai.google.dev/gemini-api/docs/live-api/session-management), [Vertex Live API reference](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/multimodal-live).

The ordinary Vertex `streamGenerateContent` API publicly documents online streamed chunks but no interaction/response ID plus replay cursor. Therefore no replay support should be claimed for that API based on current public documentation.

Source: [Vertex AI publisher model APIs](https://cloud.google.com/vertex-ai/generative-ai/docs/reference/rest/v1beta1/projects.locations.publishers.models).

### Amazon Bedrock

Bedrock `ConverseStream` describes a one-request event sequence (`messageStart`, content block events, `messageStop`, metadata). `InvokeModelWithResponseStream` documents stream errors as conditions for which the caller should retry the request. Neither public API documents a durable response/run ID, per-event replay cursor, `Last-Event-ID`, or retrieval endpoint that continues the original generation. Bedrock Session Management checkpoints workflow/conversation state, but that is state restoration rather than transport-event replay. `AsyncInvoke` writes a completed result to S3, which is result retrieval rather than stream replay.

Sources: [Bedrock ConverseStream](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html), [Bedrock ResponseStream](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ResponseStream.html), [Bedrock API compatibility](https://docs.aws.amazon.com/bedrock/latest/userguide/models-api-compatibility.html), [Bedrock Session Management](https://docs.aws.amazon.com/bedrock/latest/userguide/sessions.html).

## Implication for this project

"No SSE replay" is a normal, defensible POC boundary; even major providers do not uniformly provide it. If it is added later, the closest native precedent for a LangGraph-based service is LangGraph Agent Server's opt-in `stream_resumable=true` plus `Last-Event-ID`. A simpler alternative is to persist the canonical final answer and expose result retrieval after disconnect, but that is recovery of the final result, not replay of missed SSE frames.
