<!--
PR Title Format: [module] type: description
  modules: rollout, server, cli, auth, eval, misc, ci, doc
  types:   feat, fix, refactor, chore, test, doc
  stacked: prefix the whole title with [N/M]

Examples:
  [rollout] feat: add streaming support for chat completions
  [BREAKING][rollout] refactor: change Grader.grade signature
  [2/5][rollout] fix: surface harbor agent failures
  [ci] chore: update GitHub Actions versions
-->

## What

<!-- Brief description of the changes. What does this PR do? -->

## Why

<!-- Why are these changes needed? Link any related issues. -->
<!-- Closes #<issue_number> -->

## How to Test

<!-- Describe how reviewers can verify these changes. -->

## Checklist

- [ ] PR title follows `[module] type: description` format
      (labels are derived from it automatically — no need to add them by hand)
- [ ] `ruff check .` and `ruff format --check .` pass
- [ ] `pyright osmosis_ai/` passes
- [ ] `pytest` passes (new tests added if applicable)
- [ ] Public API changes are documented
- [ ] No secrets or credentials included
