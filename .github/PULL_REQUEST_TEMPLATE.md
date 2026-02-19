<!--
PR Title Format: [module] type: description
  modules: reward, rollout, server, cli, auth, eval, misc, ci, doc
  types:   feat, fix, refactor, chore, test, doc

Examples:
  [rollout] feat: add streaming support for chat completions
  [BREAKING][reward] refactor: rename decorator parameters
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
- [ ] Appropriate labels added (e.g. `enhancement`, `bug`, `breaking`)
- [ ] `ruff check .` and `ruff format --check .` pass
- [ ] `pyright osmosis_ai/` passes
- [ ] `pytest` passes (new tests added if applicable)
- [ ] Public API changes are documented
- [ ] No secrets or credentials included
