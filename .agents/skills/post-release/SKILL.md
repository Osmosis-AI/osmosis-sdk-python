---
name: post-release
description: Check downstream work after an Osmosis Python SDK release. Triage waiting docs PRs and identify updates needed in the public osmosis-plugin and workspace-template repositories.
---

# SDK post-release

Start with a read-only assessment. Carry out follow-up changes or merges only within the user's requested scope.

## Establish what shipped

Use the requested version, or the latest stable GitHub Release if none is given. Confirm the SDK release tag and a non-yanked distribution for that exact version at `https://pypi.org/pypi/osmosis-ai/<version>/json`. A merged release PR or a GitHub Release alone does not establish that users can install the package. If publication is missing or cannot be verified, report the blocker and leave dependent work pending. For an RC, report impact separately and leave stable-facing changes pending unless prerelease guidance was requested.

Use the [release comparison rules](../prepare-release-pr/SKILL.md#2-select-the-comparison-tag) and compare the previous tag to the **published tag**, even when main has moved on. Use the changelog to find user-visible changes; verify behavior, including reverts and backports, in the tagged code. Check downstream repositories against their current remote default branches and existing PRs, following each repository's own instructions.

## Check the three downstream repositories

1. **`Osmosis-AI/docs` — reconcile the PR queue.** Inspect open SDK-related PRs, including Mintlify bot and human PRs. Follow source PR links in the body when present; use the actual docs diff to match claims to shipped behavior. Check for missing coverage as well as overlap with merged or still-open PRs. Recommend **ready to merge** only when the whole diff is accurate for users, the English and Chinese pages agree, and required checks and reviews are satisfied. If a PR mixes shipped and unreleased claims, identify what to revise or split. If a feature or platform dependency is not available yet, say what it is waiting for. Leave unrelated PRs alone.
2. **`Osmosis-AI/osmosis-plugin` — check what the agent tells users to do.** Trace affected workflows through the public plugin's skills, references and bundled examples. Propose a change when following the guidance would give a wrong result or when a shipped capability materially improves an existing task. If the skill already delegates a detail to installed CLI help or current docs, verify that path before duplicating it. The plugin has its own version; an SDK release alone does not require a plugin bump.
3. **`Osmosis-AI/workspace-template` — check what a new user receives.** Inspect the shared template's starter rollouts, dependency constraints, sample configs and workspace guidance touched by the release. Check whether affected examples still install and work with the published SDK; use the smallest relevant local check and distinguish source inspection from actual execution. Update a version floor only when the template needs a newer SDK feature. Environment-specific workspace clones are outside this workflow.

## Hand off or act

Report the target version and comparison range, then one table: repository or PR, decision, evidence, next action. Account for all three repositories, including a concrete reason for “no change” or the missing evidence for “unverified.” Link existing PRs; prefer updating one that covers the work to opening a duplicate. When asked to merge, recheck the PR head, checks and review state before acting, and report the actual outcome.
