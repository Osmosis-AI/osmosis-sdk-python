# PR #270 dependency modularization review findings

> Historical review record for PR #270, reviewed against `main` at `3acde023` and the PR head at `0efad19f` on August 7, 2026. Current code and developer documentation remain authoritative.

## Summary

| ID | Finding | Disposition |
|----|---------|-------------|
| F1 | `LocalBackend` discarded the newly public `AgentWorkflow.run()` return value | Fixed in PR #270 |
| F2 | `AgentWorkflowOutput.info` was documented as grader-facing without a propagation path | Contract narrowed in PR #270 |
| F3 | `AgentWorkflowOutput` advertised multiple samples although one rollout carries one sample | Fixed in PR #270 |
| F4 | Extras-aware submit preflight can skip entrypoint existence and syntax validation | Follow-up after merge |
| F5 | Non-finite controller per-call metrics can prevent the whole trajectory from being saved | Follow-up after merge |
| F6 | Non-finite workflow output metrics can break `ContainerResult` JSON round-tripping | Follow-up after merge |
| F7 | Unknown `AgentWorkflowOutput` fields are silently ignored and can suppress ambient fallback | Follow-up after merge |

## Resolved in PR #270

### F1 — `LocalBackend` discarded explicit workflow output

- **Reproduction:** Return an [`AgentWorkflowOutput`](../../osmosis_ai/rollout/types/output.py) or bare message list from `AgentWorkflow.run()` without relying on an ambient sample source; [`LocalBackend`](../../osmosis_ai/rollout/backend/local/backend.py) previously reported success with `sample=None`.
- **Impact:** The public workflow return contract behaved differently between Local and Harbor execution, and a local grader could not grade the returned conversation.
- **Resolution:** `LocalBackend` now normalizes the return value, projects its one message history, request label, and metrics into `RolloutSample`, and consults the ambient `RolloutContext` only when the workflow returns `None`.
- **Verification:** `tests/unit/rollout/test_local_backend.py` covers explicit `AgentWorkflowOutput`, bare message-list returns, precedence over an ambient sample, malformed post-validation output, and grading of the projected sample.

### F2 — `info` was incorrectly described as grader-facing

- **Reproduction:** Return `AgentWorkflowOutput(info={...})`; the container and Harbor projections create `RolloutSample` from messages and metrics only, while `GraderContext` has no workflow-info field.
- **Impact:** The documentation promised grader-visible data that no backend delivered.
- **Resolution:** The public contract now marks `info` as reserved workflow metadata and explicitly states that current backends do not pass it to graders. Propagation can be designed separately if a concrete consumer is added.

### F3 — The public output contract advertised multiple samples

- **Reproduction:** Construct an output with two entries in `samples`; the container runner rejects it because the rollout protocol carries exactly one sample.
- **Impact:** The public model and documentation described behavior that the execution and grading protocol cannot represent.
- **Resolution:** The model now validates that `samples` contains at most one entry, while retaining the named mapping used by the container wire contract. Documentation consistently describes a single-sample return.
- **Verification:** Model tests reject two entries, and the container runner retains a defensive check for an output mutated after validation.

## Follow-up after merge

### F4 — Extras preflight can skip entrypoint validation

- **Reproduction:** Give a rollout an unsatisfied extra dependency and an entrypoint that is missing or contains invalid Python syntax; [`validate_rollout_backend`](../../osmosis_ai/platform/cli/workspace_directory_contract.py) returns a dependency warning before loading or compiling the entrypoint.
- **Impact:** A structurally broken rollout can pass local submit preflight and fail only after server-side dependency installation.
- **Recommended fix:** Always validate entrypoint existence and compile its source before the dependency gate, while continuing to skip imports that require unavailable dependencies.
- **Acceptance criterion:** Tests cover both a missing entrypoint and a syntax error when an extra dependency is unsatisfied.

### F5 — Non-finite controller metrics can drop the trajectory

- **Reproduction:** Supply `NaN` or infinity in matched per-call `cost_usd` or `logprobs`; the controller report model accepts it, the strict ATIF model rejects it during conversion, and the best-effort save wrapper leaves no trajectory file.
- **Impact:** Bad optional telemetry can discard an otherwise valid training transcript.
- **Recommended fix:** Reject non-finite values at the report-model boundary or isolate per-entry validation so the transcript is still saved without the invalid telemetry.
- **Acceptance criterion:** A trajectory with valid messages is saved when one reported metric is non-finite, and the invalid telemetry is logged or omitted.

### F6 — Non-finite output metrics break result round-tripping

- **Reproduction:** Put `NaN` or infinity in `AgentWorkflowOutput.metrics`; Pydantic JSON serialization writes `null`, after which [`ContainerResult.read`](../../osmosis_ai/rollout/container/files.py) rejects the value as a float.
- **Impact:** A successful container workflow can be reported by Harbor as an agent failure with no useful underlying result.
- **Recommended fix:** Reject non-finite output metrics at model construction and revalidate model instances when they cross the `ContainerResult` boundary.
- **Acceptance criterion:** Construction rejects `NaN` and both infinities, mutation is caught at the container boundary, and valid metrics still complete a JSON write/read round-trip.

### F7 — Unknown output fields can silently suppress fallback

- **Reproduction:** Construct `AgentWorkflowOutput(sample=..., metric=...)`; Pydantic's default `extra="ignore"` discards both misspelled fields, producing a non-`None` empty output that prevents the ambient fallback.
- **Impact:** A field typo can turn a valid workflow into a successful execution with no sample and no actionable validation error.
- **Recommended fix:** Configure the public output model with `extra="forbid"` and add tests for common singular-field misspellings.
- **Acceptance criterion:** Unknown top-level fields raise an `extra_forbidden` validation error and correctly spelled output still follows the existing container path.

## Additional observations

- The merge commit `ff354893` did not contain a merge-specific high-priority defect in the inspected conflict resolutions. Its ATIF ownership split, Harbor V2 lazy facade, optional dependency placement, and isolated wheel smoke tests were semantically appropriate.
- A macOS ARM installation failure involving the accepted LiteLLM version range was reproducible from both the PR and its base, so it is an upstream or baseline dependency risk rather than a PR #270 regression.
- Host-side extras can still enter task bundles through rollout-project dependency declarations, but that bundling behavior predates PR #270 and should be tracked separately from dependency modularization.
