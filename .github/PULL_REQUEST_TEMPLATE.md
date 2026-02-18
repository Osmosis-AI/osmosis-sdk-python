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

## Type of Change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)
- [ ] Refactor (code change that neither fixes a bug nor adds a feature)
- [ ] Documentation update
- [ ] CI/build configuration change

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
