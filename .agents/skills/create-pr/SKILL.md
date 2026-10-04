---
name: create-pr
description: 'Create a GitHub pull request for this repository with a title and body that follow the repo''s PULL_REQUEST_TEMPLATE.md conventions. Use when the user says "create a PR", "open a PR", "submit a PR", "make a pull request", or asks to push the current branch as a PR. Enforces the `[module] type: description` title format with 1-3 module brackets, and the What/Why/How to Test/Checklist body.'
---

# Create Pull Request (osmosis-sdk-python)

Create or update a PR or its description using the repository's current conventions. Use this skill for PR creation, submission, or body drafting; a commit-message-only request does not need it.

## Authoritative references

- Titles, stack conventions, and automatic labels: [CONTRIBUTING.md — Pull Requests](../../../CONTRIBUTING.md#pull-requests).
- Body sections and checklist: [.github/PULL_REQUEST_TEMPLATE.md](../../../.github/PULL_REQUEST_TEMPLATE.md). Read the current file instead of using a copied checklist.
- Checks and their scope: [CONTRIBUTING.md — Verification](../../../CONTRIBUTING.md#verification). For dependency changes, also follow [Dependency changes](../../../CONTRIBUTING.md#dependency-changes).

## Workflow

### 1. Establish scope and authorization

Inspect branch name, status, upstream, and the proposed base/head. Batch independent read-only calls. Start with `git diff --stat` and `git diff --name-only` for the full PR range, then read relevant diffs across all commits; inspect staged and unstaged changes separately. Narrow filenames or symbols if output is truncated.

Preserve unrelated changes. Commit and push only with the user's explicit authorization for the current task; reuse authorization already given. If publication needs a missing authorization, finish the reviewable local work and ask for that authorization. Force-push requires explicit authorization. Never stage or commit gitignored files unless the user explicitly names and authorizes those files.

### 2. Determine base and stack

Use the requested base, otherwise the repository's default branch (`main` here). For a stack, each PR targets the branch directly below it and only the bottom PR targets the default branch. Follow the stack title and GitHub linking instructions in [CONTRIBUTING.md](../../../CONTRIBUTING.md#pr-title-format); a title prefix alone does not create the stack.

### 3. Draft title and body

Classify the entire diff using the allowed modules/types in the contribution guide. Select up to three meaningful modules, or `misc` for broader changes; mark breaking changes and stack positions as documented there. Write a short imperative description in lowercase, without a trailing period.

Read the current PR template and fill every section. Explain the concrete behavior change and why it is needed, with enough context for a reviewer who has not seen the conversation. Keep `What` scannable and `Why` focused on motivation. `How to Test` must contain runnable commands and actual results or limitations. Copy the current checklist and tick only items supported by completed checks.

Keep each prose paragraph and list item on one physical line. Omit template HTML comments, AI attribution/footer text, and auto-generated review summaries from the authored body. For an existing PR whose scope changed, rewrite the title and body around the final implementation.

### 4. Verify

Run the checks applicable to the change from [Verification](../../../CONTRIBUTING.md#verification), including the linked type-checking guidance. Record commands and outcomes. Leave unverified checklist items unchecked and explain the limitation; do not substitute ordinary Pyright for the documented CI type checks.

### 5. Publish within authorization

For an authorized push, use `git push -u origin HEAD` only when establishing the upstream; preserve the existing push destination on subsequent pushes.

Pass multiline PR bodies as structured tool arguments. With `gh`, save the exact Markdown in a temporary file and use `--body-file`:

```bash
gh pr create --base main --title '[cli] fix: describe the actual change' --body-file /tmp/sdk-pr-body.md
```

Replace the example values and use `gh pr edit --body-file` for updates. Let the title/draft automation manage its labels as described in [Labels](../../../CONTRIBUTING.md#labels); apply other labels only when requested.

### 6. Verify and report

Read back the title, base, and body. Remove any tooling-added AI footer from the authored description, then return the PR URL and validation status.
