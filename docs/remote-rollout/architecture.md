# Architecture

The Remote Rollout system separates **LLM inference** (on training cluster) from **agent logic** (on your RolloutServer).

## System Overview

```
┌─────────────┐                      ┌─────────────────┐
│  TrainGate  │ ◄──── HTTP ────────► │  RolloutServer  │
│  (Trainer)  │                      │  (Your Agent)   │
└─────────────┘                      └─────────────────┘
```

- **TrainGate**: The Osmosis training infrastructure that provides LLM generation and collects training data
- **RolloutServer**: Your server running the agent logic (using this SDK)

## Communication Protocol

### 1. Initialization Flow

```
TrainGate                           RolloutServer
    │                                     │
    │  POST /v1/rollout/init              │
    │  {rollout_id, server_url,           │
    │   messages, completion_params}      │
    ├────────────────────────────────────►│
    │                                     │
    │          202 Accepted               │
    │  {rollout_id, tools: [...]}         │
    │◄────────────────────────────────────┤
    │                                     │
```

TrainGate sends a `RolloutRequest` to `/v1/rollout/init`. RolloutServer:
1. Calls `agent_loop.get_tools(request)` to get available tools
2. Returns 202 Accepted with `InitResponse` containing tools
3. Starts the agent loop in a background task

### 2. Agent Loop Execution

```
RolloutServer                       TrainGate
    │                                   │
    │  POST /v1/chat/completions        │
    │  {rollout_id, messages, ...}      │
    ├──────────────────────────────────►│
    │                                   │
    │       LLM Response                │
    │  {choices, usage, token_ids}      │
    │◄──────────────────────────────────┤
    │                                   │
    │  (Execute tools locally)          │
    │                                   │
    │  POST /v1/chat/completions        │
    │  (with tool results appended)     │
    ├──────────────────────────────────►│
    │                                   │
    │  ... repeat until done ...        │
    │                                   │
```

Key points:
- RolloutServer calls TrainGate's `/v1/chat/completions` for LLM generation
- Messages are append-only (never modify previous messages)
- The `rollout_id` routes requests to the correct training session

### 3. Completion Notification

```
RolloutServer                       TrainGate
    │                                   │
    │  POST /v1/rollout/completed       │
    │  {rollout_id, status,             │
    │   final_messages, metrics}        │
    ├──────────────────────────────────►│
    │                                   │
    │          200 OK                   │
    │◄──────────────────────────────────┤
    │                                   │
```

## See Also

- [Agent Loop Guide](./agent-loop.md) -- Complete API documentation
- [Examples](./examples.md) -- Working code examples
