---
name: create-pr
description: 'Create a GitHub pull request for this repository with a title and body that follow the repo''s PULL_REQUEST_TEMPLATE.md conventions. Use when the user says "create a PR", "open a PR", "submit a PR", "make a pull request", or asks to push the current branch as a PR. Enforces the `[module] type: description` title format with 1-3 module brackets, and the What/Why/How to Test/Checklist body.'
---

# Create Pull Request (osmosis-sdk-python)

This skill encodes the PR conventions for the `osmosis-sdk-python` repo so PRs land with a title and body that match `.github/PULL_REQUEST_TEMPLATE.md` and pass the checklist items.

## When To Use

Trigger this skill when the user asks to:
- Create / open / submit a PR (or "pull request")
- Push the current branch as a PR
- Draft a PR description for the current changes

Skip this skill if the user only wants a commit message or is not on a feature branch.

## Title Format

All PR titles MUST follow:

```
[module] type: description
```

With optional `[BREAKING]` prefix when the change is backward-incompatible:

```
[BREAKING][module] type: description
```

For multi-module changes, the title may include **1-3 module brackets total**:

```
[mod1][mod2] type: description
[BREAKING][mod1][mod2][mod3] type: description
```

For a PR that is one link in a stack, prefix the **whole** title with its position, ahead of `[BREAKING]`:

```
[N/M][module] type: description
[N/M][BREAKING][mod1][mod2] type: description
```

Number the stack bottom-to-top, so `[1/M]` is the PR based on `main` and `[M/M]` sits at the top. Use it only for a genuine stack (each PR based on the one below it); a lone PR takes no prefix.

### Allowed modules

`rollout`, `server`, `cli`, `auth`, `eval`, `misc`, `ci`, `doc`

Pick the module that best matches the primary area of change. If the diff touches multiple modules meaningfully, stack **up to three** brackets in descending importance: `[cli][rollout] ...`.

If the diff spans **more than three modules** or is truly cross-cutting / repo-wide (runtime floor changes, repo tooling, broad modernization), collapse the title to `[misc]` instead of exceeding the bracket limit.

### Allowed types

`feat`, `fix`, `refactor`, `chore`, `test`, `doc`

Type is lowercase. The `description` is a short imperative phrase, lowercase, no trailing period.

### Title examples

Good:
- `[rollout] feat: add streaming support for chat completions`
- `[cli] refactor: replace ID-based commands with name/path identifiers`
- `[BREAKING][rollout] refactor: rename rollout_v2 package to rollout`
- `[BREAKING][misc] chore: align codebase with Python 3.12`
- `[ci] chore: update GitHub Actions versions`
- `[eval] fix: handle empty rubric response`
- `[2/5][rollout] fix: surface harbor agent failures` (second of a five-PR stack)
- `[1/2][BREAKING][rollout] refactor: remove the legacy Harbor backend` (stack position precedes `[BREAKING]`)

Bad:
- `Add streaming support` (no module/type)
- `[rollout] Feat: Add streaming support.` (capitalized type, trailing period)
- `feat(rollout): add streaming` (conventional-commits style; this repo uses bracket style)
- `[cli][auth][eval][rollout] chore: align codebase with Python 3.12` (too many module brackets)
- `[BREAKING][1/2][rollout] refactor: drop the old backend` (stack position must come first)
- `[1/2] feat: add a thing` (stack position does not replace the module bracket)

## Body Template

Fill in every section. Keep it concise and specific to the diff — do NOT keep the HTML comments in the final body.

```markdown
## What

- <Bulleted, scannable list of the concrete changes in this PR.>
- <One bullet per meaningful change; reference files/modules when useful.>

## Why

<Short prose explaining motivation and context. Link related issues with "Closes #123" when applicable. If this is a breaking change, call it out here and explain the migration.>

## How to Test

- <Concrete commands reviewers can run, e.g. `uv run pytest tests/unit/cli/`>
- <Manual steps or CLI invocations to exercise the change>
- <Any `rg` / grep checks that prove an invariant (e.g. "rg 'rollout_v2' . returns no matches")>

## Checklist

- [x] PR title follows `[module] type: description` format
      (labels are derived from it automatically — no need to add them by hand)
- [x] `ruff check .` and `ruff format --check .` pass
- [x] `pyright osmosis_ai/` passes
- [x] `pytest` passes (new tests added if applicable)
- [x] Public API changes are documented
- [x] No secrets or credentials included
```

### Rules for the body

- **One paragraph per line.** Do not hard-wrap prose. Each paragraph and each list item is a single line and is allowed to soft-wrap; only semantically meaningful line breaks (a new paragraph, a new list item, code, or table rows) get an actual newline. This matches the repo invariant in `AGENTS.md` and applies to commit messages and PR descriptions too.
- `## What` uses bullets, not prose. Each bullet is a concrete change.
- `## Why` is prose (1–3 short paragraphs). Explain motivation, not mechanics.
- `## How to Test` MUST include runnable commands. Prefer `uv run pytest ...`, `uv run ruff ...`, `uv run pyright ...`, or CLI invocations.
- `## Checklist`: tick only boxes that are actually true. If a check was not run, leave it `[ ]` — do not fabricate results.
- Do NOT include AI attribution/footer text such as `Made with [Cursor](https://cursor.com)`, `Made with Claude`, or any auto-generated "Summary by cubic" section. Those are added by tooling, not by the author.
- If GitHub CLI or local tooling appends an AI footer like `Made with [Cursor](https://cursor.com)` or `Made with Claude`, immediately remove it with `gh pr edit --body ...`.

## Labels

**Do not apply type, module, `breaking`, or `stacked-pr` labels.** The `auto-label-pr` workflow derives all of them from the title, and `wip` from draft state. Do not pass `--label` to `gh pr create` and do not follow up with `gh pr edit --add-label`.

This is not merely redundant — it breaks the workflow. Reconciliation removes only labels it applied itself, identified by their `github-actions[bot]` timeline attribution. A label applied by you is attributed to you and is therefore treated as a deliberate manual override that survives every later title change. Applying the labels by hand at creation time permanently pins the PR to whatever the title said on day one.

Labels the workflow never touches, and which you may set when the user asks for them:

| Category | Labels |
|----------|--------|
| Triage | `priority: high`, `good first issue`, `help wanted`, `question`, `duplicate`, `invalid`, `wontfix` |
| Other | `dependencies`, `reward` |

For reference, what the title produces: `feat:` → `enhancement`, `fix:` → `bug`, `doc:`/`[doc]` → `documentation`, `refactor:` → `refactor`, `chore:`/`test:` → `chore`, `[BREAKING]` → `breaking`, `[N/M]` → `stacked-pr`, and `[rollout]`/`[server]`/`[cli]`/`[auth]`/`[eval]`/`[ci]` → the matching module label. `[misc]` produces none.

If the labels look wrong after the workflow runs, fix the **title** — that is the input.

## Workflow

Follow these steps in order. Batch read-only git/gh calls in parallel where possible.

### 1. Inspect the branch state (parallel reads)

Run in parallel:
- `git status` — untracked / uncommitted files
- `git diff` — unstaged changes
- `git diff --staged` — staged changes
- `git log --oneline origin/main..HEAD` (or the base branch) — commits included in this PR
- `git diff origin/main...HEAD --stat` — full diff vs. base
- `git rev-parse --abbrev-ref HEAD` — current branch
- `git rev-parse --abbrev-ref --symbolic-full-name @{u} 2>/dev/null` — whether branch has an upstream

If there are uncommitted changes, STOP and ask the user whether to commit them first. Do not auto-commit.

### 2. Determine base branch

Default base is `main`. If the user specifies otherwise (e.g. a release branch) or if `gh repo view --json defaultBranchRef` returns a different default, use that.

For a stack, each PR's base is the branch below it, and only the bottom one targets `main`. Pointing the bases at each other is necessary but not sufficient: GitHub also models the stack as its own object, which drives the stack navigation UI and rebases the remaining bases when one link merges. Create that with the `gh stack` extension (`github/gh-stack`) rather than by hand.

`gh stack submit` is the primary path but opens an interactive editor. When that is unavailable — a non-interactive shell, or PRs that already exist — use `gh stack link`, which takes PR numbers or branch names bottom-to-top, needs no local tracking state, and reuses any PRs that are already open:

```bash
gh stack link 291 292        # existing PRs, bottom first
gh stack link feat/a feat/b  # or branch names; missing PRs are created
```

Also give each PR an `[N/M]` title prefix (see Title Format above).

### 3. Classify the change

Look at the diff across ALL commits in the branch (not just HEAD) and decide:
- **Module**: which top-level area of `osmosis_ai/` changed most (`rollout/`, `cli/`, `eval/`, `platform/auth/` → `auth`, etc.). Map `platform/api/` and `platform/cli/` → `cli`. Map `rollout/server/` → `server` when the server is the primary focus, otherwise `rollout`. Choose at most **three** dominant modules; if four or more areas are meaningfully involved, or the change is repo-wide, use `misc`.
- **Type**: `feat` (new behavior), `fix` (bug), `refactor` (no behavior change), `chore` (tooling/deps/housekeeping), `test` (tests only), `doc` (docs only).
- **Breaking?**: removed/renamed public API, changed signatures, changed CLI flags users depend on, renamed packages. If yes, prepend `[BREAKING]`.

### 4. Run the checklist commands (best effort)

Before creating the PR, try to run:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright osmosis_ai/
uv run pytest
```

Record which passed. Only tick the corresponding checkbox in the body for commands that actually passed. If a command is not available or the user wants to skip, leave the box unchecked and mention it in `## How to Test`.

### 5. Push the branch (if needed)

```bash
git push -u origin HEAD
```

Only push with `-u` on first push. Never force-push without explicit user consent.

### 6. Create the PR

Always pass the body via HEREDOC to preserve formatting:

```bash
gh pr create \
  --base main \
  --title "[cli] feat: add train metrics export" \
  --body "$(cat <<'EOF'
## What

- Add `osmosis train metrics <name> --export csv` subcommand.
- Implement CSV writer in `osmosis_ai/platform/cli/metrics.py`.
- Add unit tests in `tests/unit/cli/test_train_metrics.py`.

## Why

Users asked for a way to pull training metrics into spreadsheets for post-hoc analysis. Closes #142.

## How to Test

- `uv run pytest tests/unit/cli/test_train_metrics.py`
- `osmosis train metrics my-run --export csv > metrics.csv`

## Checklist

- [x] PR title follows `[module] type: description` format
      (labels are derived from it automatically — no need to add them by hand)
- [x] `ruff check .` and `ruff format --check .` pass
- [x] `pyright osmosis_ai/` passes
- [x] `pytest` passes (new tests added if applicable)
- [x] Public API changes are documented
- [x] No secrets or credentials included
EOF
)"
```

### 7. Verify and report

After `gh pr create` succeeds:
- Capture the PR URL from stdout and return it to the user.
- Run `gh pr view --json body` and remove any auto-appended AI footer if present.
- Optionally run `gh pr view --web` only if the user asked to open it in a browser.

## Complete Example

Given a branch that renamed `rollout_v2` → `rollout` across the codebase, the correct invocation is:

- **Title**: `[BREAKING][rollout] refactor: rename rollout_v2 package to rollout`
- **Labels**: none passed to `gh pr create`; the workflow derives `rollout`, `refactor`, and `breaking` from that title
- **Body**: What bullets list the rename + import updates; Why explains namespace cleanup + migration note; How to Test includes `uv run pytest`, `uv run ruff check .`, `uv run pyright osmosis_ai/`, and `rg "rollout_v2" .` sanity check.

## Anti-Patterns

- Do NOT hard-wrap prose in the body — keep one paragraph / list item per line (see `AGENTS.md`).
- Do NOT use Conventional Commits syntax (`feat(rollout): ...`) — this repo uses bracket style.
- Do NOT use more than three module brackets in the title; switch to `[misc]` for broad cross-cutting changes.
- Do NOT omit `[BREAKING]` when the change removes or renames public API.
- Do NOT tick checklist boxes for checks you didn't actually run.
- Do NOT include AI attribution/footer text such as `Made with [Cursor](https://cursor.com)`, `Made with Claude`, or auto-generated sections like `Summary by cubic` in the body you author.
- Do NOT force-push or amend already-pushed commits without explicit user approval.
- Do NOT invent labels that don't exist in `gh label list`.
- Do NOT pass `--label` to `gh pr create` or run `gh pr edit --add-label` for type/module/`breaking`/`stacked-pr` labels — a manually applied label is pinned forever and stops tracking the title.
