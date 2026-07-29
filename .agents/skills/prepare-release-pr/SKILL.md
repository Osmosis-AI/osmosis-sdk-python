---
name: prepare-release-pr
description: Prepare an Osmosis AI Python SDK release pull request by selecting the correct stable or release-candidate comparison tag, curating concise changelog notes from the actual GitHub and git changes, updating the package version, and running release checks. Use when asked to prepare, draft, or update an RC or stable release PR.
---

# Prepare Release PR

Prepare the version bump and `CHANGELOG.md` entry for an SDK release. Inspect the underlying changes and write the changelog directly in clear language.

## Guardrails

- Read `AGENTS.md` before making changes.
- Prepare local changes only unless the user explicitly asks to commit, push, or open the PR.
- Preserve unrelated work. If the worktree contains overlapping changes, stop and explain the conflict before editing.
- Never create or move a release tag as part of this skill.
- Never infer that an RC should become the latest stable release.

## 1. Establish the Target

Accept release versions as `X.Y.Z` or `X.Y.ZrcN`, without a leading `v` in source files. Tags use the same version with a leading `v`.

Run the following preflight checks:

```bash
VERSION=0.3.0rc1  # Replace with the target version.
TARGET_REF=origin/main
git status --short --branch
git fetch origin --tags
git tag --list --sort=-version:refname
gh release view "v$VERSION"
```

Default `TARGET_REF` to `origin/main`. The `gh release view` command must report that the target release does not exist; stop if it finds one. Also confirm that the target tag does not exist locally. If the requested version is ambiguous, ask the user to specify it before editing files.

## 2. Select the Comparison Tag

Consider only published version tags reachable from the target commit. Apply these rules deterministically:

- Stable target (`X.Y.Z`): select the newest earlier stable tag and ignore every prerelease tag, including RCs for this same version.
- RC target (`X.Y.ZrcN`): select the newest earlier RC for the same `X.Y.Z` release line. If none exists, select the newest earlier stable tag.
- Never compare an RC against an RC from a different release line.
- Never compare a stable release only against its last RC; stable notes must cover the complete change since the prior stable release.

Examples:

| Target | Comparison tag |
| --- | --- |
| `0.3.0rc1` | `v0.2.30` |
| `0.3.0rc2` | `v0.3.0rc1` |
| `0.3.0` | `v0.2.30` |
| `0.4.0rc1` | newest stable `v0.3.x` tag |

Report the selected comparison tag before changing files so the choice is visible and auditable.

Set `PREVIOUS_TAG` to the selected tag and confirm that it is a published GitHub release with `gh release view "$PREVIOUS_TAG"`.

## 3. Build a Release Inventory

Inspect the complete comparison range rather than relying on commit subjects alone:

```bash
git log --oneline "$PREVIOUS_TAG..$TARGET_REF"
git diff --stat "$PREVIOUS_TAG..$TARGET_REF"
```

Use the PR numbers in the commit log to inspect associated PR titles, labels, descriptions, and relevant diffs with `gh pr view`. Verify unclear PR titles against their actual behavior. Include only changes contained in the selected range; do not ask GitHub to generate the changelog text.

## 4. Write the Changelog

Write for SDK users, not repository maintainers:

- Put breaking changes first and state the required migration directly.
- Use only non-empty headings such as `Breaking Changes`, `Added`, `Changed`, and `Fixed`.
- Describe outcomes and API or CLI behavior, not internal implementation details.
- Link each item to its PR when one exists.
- Consolidate closely related PRs into one bullet.
- Omit version bumps, formatting, tests, CI refactors, dependency housekeeping, and other internal-only noise unless users must act on them.
- Keep each bullet to one short sentence whenever possible. Prefer roughly 3–8 meaningful bullets; exceed that only when omitting items would hide material changes.
- Do not copy raw PR titles when a shorter, clearer summary is possible.
- Do not add author credits to every bullet; the linked PR already preserves attribution.
- End the entry with a full comparison link from the selected tag to the target tag.

Keep the changelog concise. It is a release summary, not a narrative of every merged PR.

Avoid unnecessary line breaks. Keep each paragraph and each list item on one physical line and allow the renderer to soft-wrap it. Insert line breaks only for a new heading, paragraph, list item, code block, or table row. Never hard-wrap prose to a fixed column width.

RC entries describe only the delta from the selected prior RC or stable release. Stable entries describe the complete delta from the prior stable release. When preparing a stable release, replace duplicated detail in same-version RC entries with short links to the corresponding RC GitHub releases so the changelog does not repeat the same material several times.

## 5. Apply the Release Changes

Update `PACKAGE_VERSION` in `osmosis_ai/consts.py` and prepend the versioned entry to `CHANGELOG.md` directly. If an entry for the target version already exists, revise it in place instead of creating a duplicate. Do not add a generator script or generated-notes workflow for these two small edits.

Review the resulting diff manually. Confirm that:

- `PACKAGE_VERSION` exactly matches the target without the `v` prefix.
- The newest changelog heading contains the target version and intended release date.
- Every changelog claim is supported by a PR or diff in the selected range.
- The comparison URL uses the selected previous tag and target tag.
- Prose and bullets are not hard-wrapped.

Use `[misc] chore: bump version to <version>` as the default PR title. If the user asks to open the PR, follow the repository's `create-pr` skill after these changes are ready.

## 6. Verify and Hand Off

Run the focused checks first, then the repository gates:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright osmosis_ai/
uv run pytest
git diff --check
```

Report the target version, comparison tag, changelog scope, files changed, and checks run. Call out any intentionally omitted internal changes or checks that did not run.
