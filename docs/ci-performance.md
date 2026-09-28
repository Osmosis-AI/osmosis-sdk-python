# CI performance investigation (2026-09-28)

Tracking: [OSM-1950 — Reduce SDK PR CI latency without weakening validation](https://linear.app/osmosis-ai/issue/OSM-1950/reduce-sdk-pr-ci-latency-without-weakening-validation).

The measured bottleneck is pytest, especially Python 3.12 coverage. Use four pytest-xdist workers to target that path; use ty as the primary source checker for faster type feedback while retaining Pyright's public API completeness gate. No GitHub runner performance improvement has been measured for this patch yet. The working tree was clean before `git pull --ff-only origin main`; the base is `2ec76ec3f2df1071f1a778a068000f032fe52566`.

## Candidate and preserved checks

- Run the complete Python 3.12/3.13/3.14 suites with `-n 4 --dist worksteal`. Keep the existing coverage flags, 70% threshold, pytest-timeout settings and fail-fast policy. Local pytest remains serial unless requested.
- Pin ty 0.0.84 in the dev group, check all of `osmosis_ai/` with warnings treated as failures, and target Python 3.12. Fix the 11 migration diagnostics with local annotations, explicit None narrowing and two narrow casts across four files. No ignored rules or source exclusions were added. Keep Pyright 1.1.411 for wheel-installed public API verification and comparison.
- Preserve the `typecheck-pyright` job ID: the live [CI Required ruleset](https://github.com/Osmosis-AI/osmosis-sdk-python/rules/12886440) requires `lint`, `typecheck-pyright`, `pytest (3.12)`, `pytest (3.13)`, `build` and `check-title`. Python 3.14 and all wheel smoke jobs continue running even though this ruleset does not list them.
- Preserve the existing bare/server/strands/openai-agents/harbor/rubric/parquet/eval/full matrix verbatim: every scenario gets a separate venv, actual wheel installation, dependency consistency check, isolated imports outside the checkout, and present/absent-package assertions.
- Leave cache policy and wheel job topology unchanged: their measured contribution does not justify changing them to shorten this critical path. [setup-uv v7 prunes downloaded wheels by default](https://github.com/astral-sh/setup-uv/blob/v7/docs/caching.md); restoring a cache is not the same as restoring a complete environment.

## GitHub Actions historical baseline

Collected 2026-09-28 from GitHub API: the latest 30 Tests pull-request runs, 2026-09-24 23:10:37 UTC through 2026-09-28 08:40:43 UTC, across 11 PRs. Outcomes: 23 success, 1 failure, 6 cancelled. Successful-run latency distributions exclude incomplete/cancelled work; those runs remain in all raw tables. P90 uses linear interpolation at (n−1)×0.9. Times are seconds and REST step timestamps have one-second granularity.

Collection caution: the event=pull_request endpoint returned a stale list ending September 5 while the unfiltered endpoint returned September 28. Selection therefore uses the latest 400 unfiltered runs and filters locally. GitHub connector jobs omit timestamps, so the full REST jobs JSON was downloaded with gh; logs were fetched with the GitHub connector. No remote run was triggered.

### Baseline distribution

| Metric | P50 / P90 / range (s) | n |
|---|---:|---:|
| Tests trigger to final job completion | 225 / 242.8 / 205–266 | 23 |
| Tests trigger to workflow updated_at | 226 / 243 / 206–266 | 23 |
| Same-trigger PR workflows to last updated_at | 226 / 263.6 / 206–308 | 23 |
| Observed Codecov completion after Tests; external origin unknown | 35 / 60.2 / 0–65 | 19 |

Same-trigger PR workflow cohort means same head SHA, event pull_request or pull_request_target, and workflow creation within ±5 seconds of Tests creation. It includes title/label/auto-approve workflows; workflow updated_at is only a completion proxy. Tests wall uses max(job.completed_at), not updated_at. 22/23 successful Tests runs finish on pytest (3.12), 1/23 on pytest (3.13); Pyright never determines completion.

| Job / step | P50 / P90 / range (s) |
|---|---:|
| Wheel smoke (bare) / Install and verify wheel | 2 / 2.8 / 1–5 |
| Wheel smoke (eval) / Install and verify wheel | 3 / 3.8 / 2–5 |
| Wheel smoke (full) / Install and verify wheel | 14 / 16.8 / 12–17 |
| Wheel smoke (harbor) / Install and verify wheel | 6 / 7 / 5–7 |
| Wheel smoke (openai-agents) / Install and verify wheel | 9 / 10 / 6–11 |
| Wheel smoke (parquet) / Install and verify wheel | 2 / 2 / 1–3 |
| Wheel smoke (rubric) / Install and verify wheel | 7 / 8 / 6–11 |
| Wheel smoke (server) / Install and verify wheel | 2 / 2 / 1–2 |
| Wheel smoke (strands) / Install and verify wheel | 8 / 8.8 / 6–10 |
| pytest (3.12) / Install dependencies | 3 / 4.8 / 2–7 |
| pytest (3.12) / Run tests with coverage | 207 / 211.6 / 170–217 |
| pytest (3.12) / Upload coverage to Codecov | 3 / 4 / 2–5 |
| pytest (3.13) / Install dependencies | 4 / 4.8 / 3–5 |
| pytest (3.13) / Run tests | 152 / 156.4 / 149–168 |
| pytest (3.14) / Install dependencies | 4 / 5 / 3–5 |
| pytest (3.14) / Run tests | 157 / 163 / 131–165 |
| typecheck-pyright / Install dependencies | 3 / 3 / 2–4 |
| typecheck-pyright / Run pyright | 9 / 10 / 6–10 |
| typecheck-pyright / Verify public API types | 4 / 4 / 3–5 |

Wheel installation and verification share one historical step. These step durations must not be represented as pure dependency-install time. Supplemental logs cover bare, harbor and full in the latest three runs (nine logs, `wheel-log-timing.csv`): full installation intervals are approximately 1.86–2.94s and verification/import intervals 9.54–12.82s. These log-derived boundaries are approximate and use a smaller sample than the 23-run step distribution.

### Waiting before execution

Separate run creation→job creation (workflow dispatch, or build dependency plus dispatch for smoke), job creation→job started (GitHub recorded runner queue), and job started→first step (runner startup gap). Public timestamps do not prove the infrastructure cause of each gap. Never attribute needs: build time to runner queue.

| Job | Dependency P50/P90/range | Dispatch P50/P90/range | Runner queue P50/P90/range | Startup gap P50/P90/range | Total before first step P50/P90/range |
|---|---:|---:|---:|---:|---:|
| typecheck-pyright | 0 / 0 / 0–0 | 1 / 23.4 / 0–42 | 2 / 3 / 1–3 | 1 / 1 / 0–1 | 4 / 26.4 / 3–45 |
| pytest (3.12) | 0 / 0 / 0–0 | 1 / 24.2 / 0–42 | 2 / 3 / 2–3 | 1 / 1 / 0–36 | 3 / 36.6 / 3–45 |
| pytest (3.13) | 0 / 0 / 0–0 | 1 / 24.2 / 0–42 | 2 / 4 / 2–71 | 1 / 1 / 0–2 | 4 / 39.2 / 3–72 |
| pytest (3.14) | 0 / 0 / 0–0 | 1 / 24.2 / 0–42 | 2 / 3 / 2–3 | 1 / 1 / 0–2 | 4 / 26.6 / 3–44 |
| build | 0 / 0 / 0–0 | 1 / 24.2 / 0–42 | 2 / 3 / 1–38 | 1 / 1 / 0–1 | 4 / 36.6 / 3–45 |
| Wheel smoke (full) | 17 / 45.4 / 11–57 | 0 / 1 / 0–1 | 2 / 2 / 2–4 | 1 / 1 / 0–2 | 20 / 49.4 / 15–60 |

Examples: run36169826784 waits 42s before initial jobs are created, then runner queue is 2s; run36378357140 pytest3.13 has a genuine 71s created→started gap; run36073551205 pytest3.12 has a 36s started→first-step gap. These are different phenomena.

### Cache evidence

For each of the 23 successful runs, downloaded complete typecheck and coverage-job logs: 46/46 report a cache hit. Every Python 3.12 cache is the same pruned key, 1,411,457 bytes, and package-download messages still appear. A cache-hit flag therefore does not mean all dependencies were restored. Dependency installation median is only 3s; changing cache policy is not the first overall-latency target. The observed key is shared by build/typecheck/wheel/pytest jobs; this is evidence of shared key usage, not proof of cache corruption.

### Balanced sensitivity check

Take only the latest successful run for each PR: n=11, Tests P50/P90/range=223 / 227 / 205–254 seconds; same-trigger workflow completion=223 / 228 / 206–254. Critical path counts: {'pytest (3.12)': 11}. This avoids over-weighting repeated #396/#397/#400 runs and preserves the same conclusion.

The one failed run, [36377070186](https://github.com/Osmosis-AI/osmosis-sdk-python/actions/runs/36377070186), failed the same two wire tool-loop tests on all three Python versions because an HTTP client had already closed. The next successful run used a different commit, `1bbe4c775f`, which retained one AsyncHTTPHandler for the fixture lifetime and closed it at teardown. This was a test-resource lifetime repair, not evidence of recovery on an unchanged rerun or a runner timeout. The six cancelled runs remain in the audit table but do not provide completed-suite latency samples.

### Per-run audit table

| PR | Run | UTC created | Outcome | Tests s | Same-trigger workflows s | Type job s | Pytest3.12 install / coverage / queue / dispatch / startup s | Codecov observed tail s |
|---:|---|---|---|---:|---:|---:|---|---:|
| 401 | [36398789006](https://github.com/Osmosis-AI/osmosis-sdk-python/actions/runs/36398789006) | 2026-09-28T08:40:43Z | success | 224 | 225 | 23 | 3.0 / 210.0 / 3.0 / 0.0 / 0.0 | 61.0 |
| 400 | [36394651204](https://github.com/Osmosis-AI/osmosis-sdk-python/actions/runs/36394651204) | 2026-09-28T07:58:53Z | success | 224 | 225 | 22 | 2.0 / 209.0 / 2.0 / 0.0 / 1.0 | 60.0 |
| 400 | [36382963488](https://github.com/Osmosis-AI/osmosis-sdk-python/actions/runs/36382963488) | 2026-09-28T05:40:56Z | success | 226 | 226 | 22 | 3.0 / 210.0 / 3.0 / 0.0 / 1.0 | 40.0 |
| 400 | [36378357140](https://github.com/Osmosis-AI/osmosis-sdk-python/actions/runs/36378357140) | 2026-09-28T04:35:19Z | success | 238 | 239 | 27 | 2.0 / 210.0 / 2.0 / 0.0 / 1.0 | 0 |
| 400 | [36377070186](https://github.com/Osmosis-AI/osmosis-sdk-python/actions/runs/36377070186) | 2026-09-28T04:17:22Z | failure | 207 | 207 | 26 | 5.0 / 189.0 / 2.0 / 1.0 / 0.0 | unobserved |
| 399 | [36189983558](https://github.com/Osmosis-AI/osmosis-sdk-python/actions/runs/36189983558) | 2026-09-25T21:09:37Z | success | 215 | 216 | 23 | 7.0 / 193.0 / 3.0 / 0.0 / 1.0 | 55.0 |
| 399 | [36189420632](https://github.com/Osmosis-AI/osmosis-sdk-python/actions/runs/36189420632) | 2026-09-25T21:03:43Z | success | 225 | 226 | 21 | 3.0 / 208.0 / 3.0 / 0.0 / 0.0 | 28.0 |
| 397 | [36170745325](https://github.com/Osmosis-AI/osmosis-sdk-python/actions/runs/36170745325) | 2026-09-25T18:01:06Z | success | 205 | 206 | 23 | 5.0 / 184.0 / 2.0 / 0.0 / 1.0 | 12.0 |
| 397 | [36170325325](https://github.com/Osmosis-AI/osmosis-sdk-python/actions/runs/36170325325) | 2026-09-25T17:57:06Z | success | 229 | 230 | 21 | 2.0 / 209.0 / 2.0 / 1.0 / 1.0 | unobserved |
| 396 | [36170316850](https://github.com/Osmosis-AI/osmosis-sdk-python/actions/runs/36170316850) | 2026-09-25T17:57:01Z | success | 221 | 222 | 19 | 2.0 / 205.0 / 2.0 / 1.0 / 1.0 | 46.0 |
| 397 | [36169826784](https://github.com/Osmosis-AI/osmosis-sdk-python/actions/runs/36169826784) | 2026-09-25T17:52:18Z | success | 266 | 266 | 24 | 2.0 / 208.0 / 2.0 / 42.0 / 1.0 | unobserved |
| 396 | [36169814265](https://github.com/Osmosis-AI/osmosis-sdk-python/actions/runs/36169814265) | 2026-09-25T17:52:11Z | success | 244 | 244 | 25 | 4.0 / 186.0 / 2.0 / 39.0 / 1.0 | unobserved |
| 397 | [36169598330](https://github.com/Osmosis-AI/osmosis-sdk-python/actions/runs/36169598330) | 2026-09-25T17:50:08Z | cancelled | 171 | 172 | 27 | 3.0 / 153.0 / 2.0 / 1.0 / 1.0 | unobserved |
| 396 | [36169584096](https://github.com/Osmosis-AI/osmosis-sdk-python/actions/runs/36169584096) | 2026-09-25T17:50:01Z | cancelled | 168 | 169 | 25 | 2.0 / 154.0 / 2.0 / 0.0 / 1.0 | unobserved |
| 397 | [36169069466](https://github.com/Osmosis-AI/osmosis-sdk-python/actions/runs/36169069466) | 2026-09-25T17:45:04Z | success | 227 | 228 | 18 | 5.0 / 183.0 / 2.0 / 25.0 / 0.0 | unobserved |
| 396 | [36169068702](https://github.com/Osmosis-AI/osmosis-sdk-python/actions/runs/36169068702) | 2026-09-25T17:45:03Z | success | 210 | 210 | 22 | 4.0 / 170.0 / 3.0 / 21.0 / 0.0 | 65.0 |
| 397 | [36168720542](https://github.com/Osmosis-AI/osmosis-sdk-python/actions/runs/36168720542) | 2026-09-25T17:41:42Z | cancelled | 226 | 227 | 22 | 4.0 / 183.0 / 2.0 / 24.0 / 1.0 | unobserved |
| 396 | [36168710771](https://github.com/Osmosis-AI/osmosis-sdk-python/actions/runs/36168710771) | 2026-09-25T17:41:36Z | cancelled | 228 | 228 | 20 | 2.0 / 151.0 / 2.0 / 60.0 / 1.0 | unobserved |
| 397 | [36168431191](https://github.com/Osmosis-AI/osmosis-sdk-python/actions/runs/36168431191) | 2026-09-25T17:38:51Z | cancelled | 194 | 195 | 24 | 2.0 / 154.0 / 27.0 / 1.0 / 1.0 | unobserved |
| 396 | [36168430885](https://github.com/Osmosis-AI/osmosis-sdk-python/actions/runs/36168430885) | 2026-09-25T17:38:51Z | cancelled | 224 | 225 | 25 | 2.0 / 210.0 / 2.0 / 1.0 / 1.0 | unobserved |
| 395 | [36168335807](https://github.com/Osmosis-AI/osmosis-sdk-python/actions/runs/36168335807) | 2026-09-25T17:37:56Z | success | 223 | 223 | 22 | 3.0 / 206.0 / 2.0 / 0.0 / 1.0 | 24.0 |
| 397 | [36167400007](https://github.com/Osmosis-AI/osmosis-sdk-python/actions/runs/36167400007) | 2026-09-25T17:29:03Z | success | 223 | 308 | 25 | 2.0 / 207.0 / 2.0 / 1.0 / 0.0 | 27.0 |
| 396 | [36167396342](https://github.com/Osmosis-AI/osmosis-sdk-python/actions/runs/36167396342) | 2026-09-25T17:29:01Z | success | 227 | 228 | 23 | 2.0 / 209.0 / 3.0 / 0.0 / 1.0 | 40.0 |
| 395 | [36167392309](https://github.com/Osmosis-AI/osmosis-sdk-python/actions/runs/36167392309) | 2026-09-25T17:28:58Z | success | 236 | 308 | 21 | 3.0 / 217.0 / 2.0 / 1.0 / 2.0 | 33.0 |
| 394 | [36097336791](https://github.com/Osmosis-AI/osmosis-sdk-python/actions/runs/36097336791) | 2026-09-25T05:08:51Z | success | 219 | 219 | 20 | 2.0 / 205.0 / 2.0 / 1.0 / 0.0 | 30.0 |
| 394 | [36096795189](https://github.com/Osmosis-AI/osmosis-sdk-python/actions/runs/36096795189) | 2026-09-25T05:01:06Z | success | 226 | 227 | 24 | 2.0 / 212.0 / 2.0 / 1.0 / 0.0 | 35.0 |
| 392 | [36073551205](https://github.com/Osmosis-AI/osmosis-sdk-python/actions/runs/36073551205) | 2026-09-24T23:36:10Z | success | 254 | 254 | 21 | 2.0 / 204.0 / 2.0 / 1.0 / 36.0 | 14.0 |
| 389 | [36073501209](https://github.com/Osmosis-AI/osmosis-sdk-python/actions/runs/36073501209) | 2026-09-24T23:35:33Z | success | 227 | 228 | 23 | 2.0 / 213.0 / 2.0 / 1.0 / 0.0 | 43.0 |
| 391 | [36071790835](https://github.com/Osmosis-AI/osmosis-sdk-python/actions/runs/36071790835) | 2026-09-24T23:15:19Z | success | 224 | 225 | 26 | 3.0 / 204.0 / 2.0 / 0.0 / 1.0 | 18.0 |
| 390 | [36071395670](https://github.com/Osmosis-AI/osmosis-sdk-python/actions/runs/36071395670) | 2026-09-24T23:10:37Z | success | 219 | 220 | 26 | 3.0 / 204.0 / 2.0 / 0.0 / 1.0 | 54.0 |

### External checks and limits

All external check/status trigger origins are unknown from the retained API payloads. Same-SHA latest completion is not automatically the waiting time caused by the PR push. Codecov is observed on 19/23 successful heads and often finishes after Tests; the observed tail median is 35s, P90 60.2s, range 0–65s. Missing Codecov observations on four heads are missing data, not zero processing time. Logs preserve coverage-upload success independently from external report completion.

A clear counterexample to indiscriminate attribution is run36382963488: Tests starts 05:40:56 UTC, but the retained cubic check starts 06:31:59 and ends 06:34:59. The same-SHA last check is 3,243s after Tests trigger, yet that does not establish 54 minutes of CI waiting; it may be a later manual review. run36378357140 likewise has a later-origin cubic record. external-checks.csv retains these records with explicit eligibility flags solely for inspection; the filtered observable_checks_wall_s field is not a valid end-to-end performance claim. The main legacy required-status-check endpoint returned 404 Required status checks not enabled; rulesets and human approvals are not reconstructed here.

### Minimum changes supported by this baseline

1. Optimize pytest execution first: evaluate a fixed small xdist worker count (for example 2) with identical collection, coverage, per-test timeout, and all three Python versions. Do not use unbounded auto workers without runner evidence. Verify resource/port isolation and preserve every test; only a same-commit GitHub runner comparison can validate a gain.
2. Evaluate ty as a faster primary checker separately. Retain Pyright --verifytypes or equivalent proven public-API completeness protection. Even eliminating the current 9s Pyright step predicts zero observed Tests critical-path improvement in this sample because it runs in parallel and finishes far earlier.
3. Preserve all nine independent wheel-install environments. Their combined path finishes well before the pytest critical path, so merging scenarios or removing extras cannot be justified as the minimum path to lower observed latency.
4. Keep cache and external Codecov latency separate. Installation changes have a low measured upper bound; external service turnaround and occasional dispatch/runner gaps require subsequent runner observations, not a Mac benchmark.

Evidence is retained locally under `/tmp/osmosis-ci-investigation/history/`. Raw evidence: recent-all-runs-page{1..4}.json, test-pr-runs-sampled.json, recent-pulls.json, jobs-{run}.json, checks-{run}.json, status-{run}.json, log-{job}.txt. Derived tables: runs.csv, jobs.csv, steps.csv, logs.csv, external-checks.csv, summary.json. Reproduce derived output with python3 analyze.py. collect.py and collect-checks.py preserve endpoint requests; connector log retrieval is represented by its exact raw log files.

### Small wheel-log spot check

The latest three successful runs, each with bare/harbor/full, give nine log-derived observations. These are approximate message-to-message intervals, not precise process timings; uv self-reported resolve/prepare/install phases are retained separately in wheel-log-timing.csv. REST combined-step timestamps remain the authoritative comparable baseline across all 23 runs.

| Run | Scenario | Combined step s | Venv-created log to installed log s | pip-check success log to verified log s |
|---|---|---:|---:|---:|
| 36398789006 | Wheel smoke (full) | 17.0 | 2.281 | 12.816 |
| 36398789006 | Wheel smoke (bare) | 1.0 | 0.242 | 0.77 |
| 36398789006 | Wheel smoke (harbor) | 5.0 | 1.523 | 3.126 |
| 36394651204 | Wheel smoke (harbor) | 6.0 | 1.447 | 3.783 |
| 36394651204 | Wheel smoke (full) | 14.0 | 2.938 | 9.536 |
| 36394651204 | Wheel smoke (bare) | 2.0 | 0.575 | 1.4 |
| 36382963488 | Wheel smoke (full) | 16.0 | 1.862 | 12.498 |
| 36382963488 | Wheel smoke (bare) | 2.0 | 0.448 | 1.396 |
| 36382963488 | Wheel smoke (harbor) | 6.0 | 1.429 | 3.862 |

The full scenario spends roughly 1.86–2.94s between venv creation and dependency installation and 9.54–12.82s between pip-check success and verification completion in this small spot check. Its install-and-verify step must not be relabelled entirely as installation.

## Local validation and type-check comparison

These results establish local correctness and feasibility; macOS process timings are not GitHub runner timings. The source and complete dependency manifests, commands, raw outputs and timing samples are retained under `/tmp/osmosis-ci-investigation/typecheck/` and `/tmp/osmosis-ci-investigation/pytest/`.

The initial checker comparison used an archive of exactly `2ec76ec3`, the same locked all-extras/dev environment plus ty 0.0.84, Python 3.12.12, identical `osmosis_ai/` scope, and the same Linux static target. Each checker received one warmup and seven alternating measured launches. Pyright's median was 2.653s (2.637–2.679s); ty's was 0.308s (0.283–0.339s), but ty reported 11 diagnostics on that baseline. A failing checker is not an accepted migration benchmark. After the targeted fixes, both checkers pass without diagnostics. Their rule semantics differ; identical input and scope do not imply identical findings. The candidate comparison below uses the same final working tree for both tools, since this patch has not been committed.

Public API verification used a non-editable installation and confirmed actual scanning, not just an exit status: 161 public modules, 3,543 exported symbols, 3,542 known types and one existing unknown Harbor symbol (99.9718%). The only two errors are the pre-existing unknown metaclass/base class of `harbor.harness_agent.OsmosisHarnessInstalledAgent`; the existing narrow workflow exception accepts them. There are no unexpected errors. Ordinary [ty checking](https://docs.astral.sh/ty/reference/cli/) does not replace [Pyright public API completeness](https://github.com/microsoft/pyright/blob/main/docs/typed-libraries.md).

The original-commit pytest comparison used the same Python 3.12.14 environment, including pytest-xdist 3.8.0 for both commands. Serial and two-worker runs each executed 3,606 identical testcase IDs: 3,605 passed and one skipped. Every file's line hits, branch conditions and missing branches matched exactly: 9,486/10,146 lines and 2,632/3,110 branches, combined coverage 91.42%. External wall time was 142.66s serial versus 65.07s with two workers. This is one local feasibility pair, not a CI speedup estimate. [pytest-cov supports combining xdist worker coverage](https://pytest-cov.readthedocs.io/en/stable/xdist.html).

Final candidate comparison used Python 3.12.12 on macOS ARM64, with both checkers explicitly targeting the same Darwin platform and Python 3.12. The checked source hash was `51eedafdc7def5fae1b07e1e2af3dd30e73a0be1d1d800b8c7164f6190d7b711` and the lockfile hash `0efc8d0bf841992ee2470e8ff51bd8fcac945586927e9b4cf1417f53196df5d1`. Both saw the same uncommitted working tree based on `2ec76ec3`, all extras and the same final locked dev environment; no inputs changed during measurement. One warmup per checker was excluded, followed by five alternating samples. Both passed every launch:

| Checker | Local median | Local range | Result |
|---|---:|---:|---|
| ty 0.0.84 | 0.305s | 0.286–0.328s | 0 diagnostics |
| Pyright 1.1.411 | 2.657s | 2.645–2.710s | 0 diagnostics |

These numbers are not a GitHub runner comparison. Full reproducibility metadata and per-launch logs are in `/tmp/osmosis-ci-investigation/final-typecheck/results.json`. No speed multiplier is applied to historical runner durations.

Initial two-worker candidate correctness checks:

- Python 3.12.14, 3.13.15 and 3.14.7: each complete two-worker suite passed 3,605 tests with the same one pre-existing optional-NumPy skip; all 3,606 testcase IDs match the serial baseline. Python 3.12 coverage remains 91.42%. The local serial/parallel feasibility pair ran serial first, so cache/order effects are not controlled; use the alternating runner protocol below for acceptance.
- All nine separate wheel installations passed installation, dependency consistency and isolated smoke verification. Build, strict Twine checks, public artifact boundaries and versioned filenames passed; the wheel's 167 Python source files match the candidate.
- Full-package ty and Pyright source checks passed. Non-editable public API verification passed the existing narrow Harbor exception. The 143 tests touching the type-compatibility edits also passed.
- Ruff lint/format, lockfile consistency, actionlint on the changed workflow and `git diff --check` passed. Independent review found no blocking issues.

The final test environments and wheel environments are outside the shared checkout. Evidence is under `/tmp/osmosis-ci-investigation/pytest/evidence/` and `/tmp/osmosis-ci-investigation/wheels/`.

## Further optimization investigation

The follow-up preserves the initial uncommitted patch and measures the remaining test path. Evidence is under `/tmp/osmosis-ci-investigation/further/`; the original local and GitHub history samples above remain unchanged.

### Remove unrelated test shutdown waiting

`test_duplicate_rollout_id_is_rejected` deliberately leaves a `BlockingBackend` running after checking the duplicate-ID response. Previously, leaving TestClient then waited for the production ten-second shutdown grace period. The test now checks the same admission and rejection responses before explicitly cancelling the admitted rollout and reading its terminal result with the original lease. Its call duration fell from about 10.01s to 0.01s locally, and all 24 admission tests pass. Dedicated tests still cover graceful shutdown order, cancellation past the drain deadline and bounded native cancellation cleanup; production timeouts are unchanged.

### Four-worker feasibility and test semantics

GitHub metadata identifies this as a public repository. [Standard public Ubuntu runners have four CPUs](https://docs.github.com/en/actions/reference/runners/github-hosted-runners), making a fixed four-worker experiment reasonable. This Mac has ten CPUs, so its scaling results cannot establish runner scaling.

An initial four-worker run passed, but one of the next two full runs failed two tunnel tests whose outer one-second deadline covered OS process startup as well as the behavior under test. The failures occurred before receipt of the URL, not in registration or exit handling. Those failures remain in the evidence and are not included as successful performance samples.

The two tests retain real shell subprocesses, stderr parsing and cleanup. They now explicitly assert URL and registration log processing, preserve the assertion that host probing must not occur, and require the precise registration-followed-by-exit-7 error. The successful-start test always stops its process in a finally block. These behavior assertions replace the total-startup one-second limit; the product's existing 30-second startup deadline and pytest timeout remain unchanged. No sleep duration or production timeout was shortened or inflated to make the benchmark pass.

The final Python 3.12.14 comparison used the same test fixes, source, locked all-extras/dev environment, branch coverage flags and interpreter. Runs were sequential in the order 2, 4, 2, 4, 4 workers. Every run executed the same 3,606 testcase IDs with 3,605 passes and the unchanged optional-NumPy skip:

| Workers | Local process wall times | Median | Result |
|---|---|---:|---|
| 2 | 60.03s, 59.09s | 59.56s | Both passed |
| 4 | 35.70s, 36.58s, 43.53s | 36.58s | All three passed |

This is a 38.6% local median reduction after the test fixes, not a GitHub runner result. The three four-worker samples have visible timing variance; do not extrapolate the ratio to the historical Actions baseline. Coverage was 91.39–91.42%, with identical statement and branch denominators. Two four-worker samples matched every reported line and branch in the two-worker runs. One did not execute `runner.py:1250–1251`, the watchdog's early return after cancellation/halt (two lines, one branch). This follows the existing worker/watchdog completion race; test results and terminal-state assertions remain unchanged. This specific difference is documented, not hidden by a new coverage exclusion or lower threshold.

Final four-worker runs on Python 3.13.15 and 3.14.7 each passed 3605 tests with the same one optional NumPy skip, in 30.12 s and 30.39 s of local wall time respectively. All seven final runs executed the same 3606-test ID multiset. Package/test source and dependency/configuration hashes were rechecked against the working tree and still matched; evidence is in `/tmp/osmosis-ci-investigation/further/final-pytest/`, including `findings.md`, `summary.json`, and `matrix-summary.json`. These supported-version runs establish local compatibility, not runner performance. The new default retains bounded parallelism and manual 0/2/4-worker controls for runner acceptance. No new dependency was needed for this follow-up.

### Coverage-engine experiment rejected for this patch

coverage.py 7.13.4 uses CTracer for branch coverage on Python 3.12; forcing sysmon there warns and falls back to CTracer. Python 3.14 supports branch tracing through SysMonitor and selects it by default. See the [versioned core configuration](https://coverage.readthedocs.io/en/7.13.4/config.html#run-core).

One same-source, same-environment Python 3.14/two-worker pair produced 58.87s with CTracer and 61.92s with SysMonitor. Both completed all tests, but three branch observations differed (`dataset.py:479`, `results.py:255` and `results.py:257`). This small sample supplies no speedup evidence and does not establish branch-report equivalence. Separately, the current required-check rules do not include `pytest (3.14)`, so moving the coverage gate there without updating the rule would weaken merge protection. Coverage stays on Python 3.12 with the same branch flags, threshold and exclusions; no repository rule is changed.

Do not disable pytest plugin autoload or consolidate isolated wheel environments for this patch: retained logs show only the expected asyncio/timeout/cov/anyio plugins plus xdist, installation is already inexpensive, and neither change is supported by critical-path evidence. The crash-resume tests retain their deliberate timing windows and real process recovery behavior.

## Continued investigation: slow tests and completion evidence

The continuation preserves the incoming uncommitted patch, with a byte-for-byte snapshot under `/tmp/osmosis-ci-investigation/continuation/before/`. Source, tests, configuration and dependency hashes still match the previously validated candidate. This follow-up changes only workflow instrumentation, the Auto approve concurrency key, and this document; it does not add another test-behavior optimization or type-checker migration.

### Remaining slow tests

Reanalyzing the five retained Python 3.12.14 runs on the same Mac and dependency environment gives the following ranges of summed JUnit testcase durations. These sums overlap under xdist and are neither workflow wall time nor CPU time; they identify inspection targets only.

| Test module | Sum of testcase durations per run (s) | Behavior to preserve |
|---|---:|---|
| `eval/local/test_runner_e2e.py` | 43.129–44.734 | Real child processes, HTTP calls, journaling and server shutdown |
| `eval/local/test_runner_crash_resume.py` | 12.318–16.805 | Durable journal, SIGKILL/SIGINT and recovery with unfinished work |
| `test_packaging.py` | 9.426–10.190 | Actual wheel builds and archive inspection |
| `eval/local/test_tunnel.py` | 6.028–7.742 | Real shell subprocess output and cleanup |

The crash-resume tests' 1.5/2.0-second workflow windows deliberately leave work in flight after durable progress. The runner's cancellation-cleanup delay and fresh-interpreter import checks also establish behavior. No additional unrelated long wait was found in these paths. The already-fixed duplicate-admission test is 0.007s in all five JUnit reports. Detailed testcase distributions and code references are in `/tmp/osmosis-ci-investigation/continuation/slow-tests/`; none proves which individual test is slow on GitHub, because the historical runner commands did not retain testcase durations.

### Minimal runner instrumentation

Both pytest commands now use built-in `--durations=30 --junitxml=pytest-results.xml`, without changing selection, scheduling, timeouts or coverage options. The duration log distinguishes setup/call/teardown; JUnit defaults to their combined testcase duration. The xdist controller writes the XML, so workers do not compete for the output file. Compare `(classname, name)` multisets, outcomes and skips rather than relying only on total counts.

Manual dispatches additionally record actual checkout HEAD, dirty state excluding the environment report itself, event/ref/SHA, lock/config hashes, exact Python/uv/distribution versions, worker count, CPU information and runner image. A per-Python, per-attempt artifact retains that environment report, JUnit and Python 3.12 coverage XML, including when the job fails after producing files. Existing setup-uv logs retain cache information. The environment-recording and artifact-upload steps run only on dispatch; measure their cost separately when comparing diagnostic workflow elapsed time with ordinary PR runs. The instrumentation itself is not a claimed speed optimization.

For a fixed candidate commit and comparable runner conditions, report speedup as `T0 / Tn` and parallel efficiency as `T0 / (n * Tn)`, using repeated pytest-step wall times for zero, two and four workers. This includes interpreter/worker startup, collection and coverage aggregation. JUnit does not retain worker IDs, so it cannot establish per-worker utilization or scheduler idle time. Keep queue, installation and final workflow/check completion outside this test-step ratio.

### Verified finish-time boundaries

A live recheck found no newer PR Tests run than `36398789006`. The same 30-run historical cohort was reanalyzed using actual job completion and original attempt 1 for every same-trigger workflow, excluding three later auxiliary-workflow reruns. A cohort shares head SHA, PR/PR-target event and creation within five seconds of Tests. The current main rules require six status contexts (`lint`, `typecheck-pyright`, `pytest (3.12)`, `pytest (3.13)`, `build`, `check-title`), plus approval and an up-to-date branch requirement. Applying today's status-context list is not reconstruction of historical mergeability. Codecov, cubic, Python 3.14 and wheel smoke are not in that required-status list; all existing jobs remain enabled.

| Metric | P50 / P90 / range (s) | n |
|---|---:|---:|
| Tests creation to final Tests job completion | 225 / 242.8 / 205–266 | 23 |
| Tests creation to all six required contexts successful in the original cohort | 225.5 / 243.4 / 205–266 | 22 |
| Tests creation to last recorded same-trigger Actions job completion | 226 / 263.6 / 205–307 | 23 |
| Same-trigger Actions completion after Tests, floored at zero | 0 / 0 / 0–84 | 23 |
| Python 3.12 pytest step end to job completion | 6 / 7 / 5–8 | 23 |
| Codecov upload step | 3 / 4 / 2–5 | 23 |
| Codecov upload step end to job completion | 3 / 3 / 2–3 | 23 |
| Observed external Codecov completion after Tests, floored at zero | 35 / 60.2 / 0–65 | 19 |

Only 16 of the 23 successful Tests cohorts have all recorded original-attempt Actions jobs successful. Completion of cancelled or failed auxiliary jobs must not be called “all green.” PR #391's original title check failed; a later title rerun succeeded at 23:16:36 UTC, before Tests finished at 23:19:03, but is kept separate from the 22 complete original-cohort required-check samples. Nonzero same-trigger Actions tails were Auto approve on PR #397 (84s, success) and #395 (68s, cancelled). The previous table's 308-second `updated_at` maximum was a workflow metadata proxy; actual job completion here is at most 307 seconds.

External Codecov origins remain unknown, with four missing observations; same-SHA reviewer reruns likewise cannot establish push-to-all-green time. The 35-second observed Codecov tail is separate from its three-second upload step. No uploader wait setting, external check, approval policy or required-check rule was removed to shorten these observations. Raw current rules, all-attempt checks, status history, annotations and per-run derived tables are retained under `/tmp/osmosis-ci-investigation/continuation/tail/`; `analyze.py` reproduces `summary.json`, `runs.csv`, `cohort-jobs.csv` and `external-observations.csv`.

### Avoid cross-PR Auto approve cancellation

The previous Auto approve key was `${{ github.workflow }}-${{ github.ref }}`. This workflow uses `pull_request_target`, whose ref is shared across PRs in the default-branch context; see [GitHub event semantics](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#pull_request_target). Different PRs therefore shared a concurrency group with `cancel-in-progress: true`. The key now uses `github.event.pull_request.number`, matching the existing title/label workflows and retaining cancellation of superseded runs for the same PR. Actor eligibility, permissions and the approval command are unchanged.

Historical evidence is concrete but does not establish a general time saving: Auto approve runs for PRs #395/#396/#397 were created at 17:28:58/17:29:01/17:29:03 UTC on September 25. The first two were cancelled. [PR #397's Auto approve run](https://github.com/Osmosis-AI/osmosis-sdk-python/actions/runs/36167400194) created its job at 17:34:03, started it two seconds later and completed at 17:34:10. [Its Tests run](https://github.com/Osmosis-AI/osmosis-sdk-python/actions/runs/36167400007) finished at 17:32:46, an observed 84-second Actions tail. The 300-second run-to-job creation gap must not be called runner queue or shell execution. The shared group explains cross-PR interference; API timestamps alone do not prove why cancellation/dispatch took that long or how much of the tail the key change will remove.

The cancelled PR #395 job's annotation explicitly states: `Canceling since a higher priority waiting request for Auto approve-refs/heads/main exists`. This confirms shared-group cancellation, independently of inferring it from timestamps; the retained response is `annotations-108178366711.json` in the tail evidence directory.

Because `pull_request_target` loads the default-branch workflow, a candidate PR does not itself execute this updated Auto approve definition. Validate the new grouping after an authorized normal merge, with overlapping runs on different PRs and superseded runs on one PR. Do not merge solely to enable the benchmark, and do not report this observed 84 seconds as an achieved speedup.

### Continuation validation

The full four-worker suites were rerun with the new duration/JUnit options in the retained isolated environments on Python 3.12.14, 3.13.15 and 3.14.7. Each passed 3,605 tests with the same one optional-NumPy skip; all 3,606 testcase multisets match the prior accepted local sample. Python 3.12 retains 9,486/10,146 lines and 2,632/3,110 branches, or 91.42% combined coverage against the unchanged 70% gate. Source/test/config/dependency hashes still match before and after these checks. These are Mac correctness runs, not new runner performance samples.

Ruff lint/format, actionlint 1.7.12 on both changed workflows and `git diff --check` pass. The lint, type-check/public-API, build and all nine independent wheel-smoke job definitions are unchanged from the incoming patch, as are the pytest matrix and installation step. Test logs, JUnit, coverage XML, source comparisons and preserved-check assertions are under `/tmp/osmosis-ci-investigation/continuation/validation/`. No additional ty/Pyright performance comparison was performed during the continuation.

## Reduce repeated integration work instead of chasing a coverage percentage

Both the local and live `main` [Codecov configuration](https://github.com/Osmosis-AI/osmosis-sdk-python/blob/main/codecov.yml) specify 70% project and patch targets, with a two-point project tolerance. The required pytest job separately enforces `fail_under = 70`. The previously reported 91.42% is measured combined line/branch coverage, not a 90% minimum. Lowering a percentage alone does not skip tests or reduce coverage-tracing work. There is already room below the observed coverage to remove low-value tests; no threshold change is needed for the confirmed redundancy below.

The remaining expensive repeated setups were reviewed against their assertions, with the following consolidations. No production code, fixture behavior, test selection, timeout, coverage exclusion, extra-install matrix or public API gate changes in this step.

| Test area | Consolidation | Work removed per Python suite |
|---|---|---|
| Runner E2E | Merge six identical default-run scenarios' reward, trajectory, log and UI-hook assertions into the full-run test; merge duplicate retry assertions; delete the misleading partial-journal test that only resumed a fully completed selection | 8 testcase entries, 10 runner calls, 9 real rollout-server starts |
| Tunnel | Check log forwarding, spawned process identity and empty-config arguments in the existing real start/stop test, with `finally` cleanup | 3 testcase entries and 3 shell-process starts |
| Packaging | Check optional grader in an existing workflow-only build; combine nested-cache staging, cache hit and source-change invalidation into one sequence | 2 testcase entries and 2 actual wheel builds |
| Public imports | Import the client first in the existing fresh-interpreter core-import test; retain public-class and unloaded-optional-module assertions under a superset of the old blocked dependencies | 1 testcase entry and 1 Python process |

This removes 14 testcase entries while retaining the distinct behavior checks. The removed partial-journal test never left missing work: it completed rows `(0, 1)` and then resumed that exact completed selection. The stronger complete-run no-op test remains, alongside genuine SIGKILL/SIGINT recovery. Error/cancellation paths, secret redaction, artifact durability, independent wheel-extra installations and fresh-interpreter dependency isolation remain covered. Independent review checked the assertion mappings and found no material loss of behavior coverage.

Large counts of cheap unit tests are not the measured target. Of the 3,499 cases outside runner E2E/crash-resume, packaging and public imports, 3,056 individually took less than 10ms in the retained Python 3.12 sample and summed to only 4.472s of overlapping testcase time. Deleting hundreds of them would provide little evidence-backed latency benefit. Prompt ESC tests, real crash windows and distinct failure scenarios are retained.

The incoming working tree is preserved under `/tmp/osmosis-ci-investigation/test-pruning/before/`. Baseline and candidate copies of the four changed test modules, exact hashes, command lines, testcase deltas, JUnit, coverage XML and process timings are under `/tmp/osmosis-ci-investigation/test-pruning/`. These local measurements assess consolidation correctness and cost only; they are not GitHub runner speedups. Runner acceptance must now compare the expected reduced testcase set and retained contracts, rather than require identity with the original 3,606-case suite.

### Repeated local consolidation validation

The complete Python 3.12.14 suite ran six times on the same Mac, interpreter, locked all-extras/dev environment, four-worker worksteal scheduler and branch-coverage command. The sequence was baseline, candidate, candidate, baseline, baseline, candidate, with no overlapping runs. Only the four frozen test-module variants changed; production source, dependencies, timeout and coverage configuration stayed identical, and hashes were verified unchanged during each measurement.

| Variant | Local process wall times (s) | Median (s) | Test results per run |
|---|---|---:|---|
| Before consolidation | 38.881, 38.222, 39.280 | 38.881 | 3,605 passed, 1 skipped |
| After consolidation | 36.148, 36.096, 37.229 | 36.148 | 3,591 passed, 1 skipped |

The local median reduction is 2.733s (7.0%), with only three observations per variant. This validates less repeated execution on this machine, not a GitHub runner or PR-completion improvement. Python 3.13.15 and 3.14.7 also pass the complete candidate suite with 3,591 passes and the unchanged optional-NumPy skip. Testcase multisets match within each variant and across the candidate Python versions; the delta is exactly the documented 14 cases, including the renamed merged packaging test.

All three baselines and the first two candidate runs match every reported covered line, branch condition and missing branch: 9,486/10,146 lines and 2,632/3,110 branches (91.42%). The third candidate reports 9,484 lines and 2,631 branches (91.39%), missing only `eval/local/runner.py:1250–1251`, the same watchdog early-return race already observed before consolidation. Existing terminal-state and cancellation tests still pass. No exclusion, threshold reduction or extra timing-sensitive test was added to force this percentage back up. The 70% pytest and Codecov targets remain unchanged because the tested simplification does not need a lower gate.

`test-pruning/analyze.py` reproduces the aggregate evidence in `evidence/summary.json`, including every sample, testcase delta, source hashes and line-level coverage differences. Ruff lint/format and independent review pass. No runner benchmark, commit or push was performed for this consolidation.

## Further test-maintenance simplification

A second audit removes another 22 redundant testcase entries across 12 files, with 282 net test lines removed and no replacement helper framework. This targets maintenance cost rather than claiming a measurable CI speedup from fast unit tests. Production behavior, supported Python versions, coverage thresholds and all installation/type gates are unchanged.

| Area | Cases removed | Protection retained |
|---|---:|---|
| CLI registration, help and rendering | 7 | Actual root/quickstart help, reserved JSON-key protection, exact plain output and running/finished checkpoint rendering already check the same behavior |
| CommandResult value construction | 5 | Actual JSON/plain/rich renderers and command consumers check the fields; positional exit-code compatibility and overflow typing tests remain |
| Platform labels and model parsing | 6 | Mixed/sorted secret scopes, personal-only labeling, status formatting and the real deploy client retain the same output/parser assertions |
| Rollout contexts and sample/packaging checks | 4 | Backend-to-workflow/grader metadata transfer and SDK sanitization remain; a test of Python's own JSON behavior is removed |

Review caught one distinct boundary inside the misleading Harbor `test_bundle_requirements_skips_extras`: its fixture has no extras, but `requirements == []` is still a useful empty-dependency assertion. That exact assertion now runs after the existing `inspect_bundle(bundle)` in `test_grader_wheel_ships_in_tests_dir`, using the same module-scoped wheel without an additional build or testcase. The real extras-filtering test in `test_packaging.py` remains. No default-value, authentication, secret-redaction, recovery, resource-cleanup or positional-API compatibility test was removed merely for being small.

The previous temporary environment was no longer available, so validation used a fresh isolated source copy with Python 3.12.11 and `uv sync --locked --all-extras --group dev`. The affected modules plus their retained protection tests passed 587 tests with one optional-NumPy skip. The complete four-worker branch-coverage suite passed 3,569 tests with that same skip, covering 9,486/10,146 lines and 2,632/3,110 branches (91.42%). The final relocated empty-dependency assertion passed its targeted test after the full run. Ruff lint/format and `git diff --check` pass. No timing comparison is made against the earlier Python 3.12.14 environment.

Exact deletion-to-retained-test mappings, original files, source hashes, JUnit and logs are under `/tmp/osmosis-ci-investigation/test-simplification/`. The existing uncommitted changes remain in place; no commit, push or remote run was performed.

## GitHub runner acceptance procedure

No commit, push, workflow dispatch, merge or repository-rule change was performed during this investigation. Publishing the branch requires the user's authorization. Runner results are still required before stating a CI performance gain.

1. Publish the reviewed patch after authorization and pin one candidate commit for the experiment. Verify the manual dispatch is available for that ref; GitHub documents [default-branch requirements for manual workflows](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow). Do not merge the optimization merely to enable an experiment if GitHub rejects dispatch before registration.
2. Run Tests with `pytest-workers=0`, `pytest-workers=2` and `pytest-workers=4`, keeping `benchmark-types=false` for overall workflow measurements. Compare zero versus four for the total parallelism change, and two versus four for the follow-up change. Run each to completion before starting the next because the existing concurrency group cancels overlapping runs on the same ref. Use at least three samples per worker count in counterbalanced order, for example `0,2,4; 4,0,2; 2,4,0`. Verify the same actual checkout HEAD, lockfile, installed dependencies, uv/Python versions, runner image and comparable cache conditions from logs and artifacts. PR checkout normally uses a merge commit, while dispatch checks out the chosen ref; identical PR head SHA alone is insufficient. More samples may be required if startup variance dominates. Do not compare a cold baseline only with a warm candidate.
3. Collect attempt-specific job/step timestamps and `pytest-*` artifacts for every run, including failures and cancellations. Compare tests, setup/install, pre-execution delays, diagnostic recording/upload overhead and final job completion separately. Recheck testcase multisets, outcomes, skips and coverage. Report P50/P90/ranges, sample sizes and test-step parallel efficiency; small-sample tail percentiles are preliminary. Use PR-triggered runs to confirm all actual checks, since dispatch alone does not reproduce title checks, external review triggers or user-facing all-green time. A same-commit worker comparison measures parallelism, not the prior admission-test fix or the separate Auto approve change.
4. Separately dispatch `benchmark-types=true`. The script runs ty and Pyright in the same job, interpreter, dependency environment, platform and source range; each gets one warmup and five alternating timed launches. The `typecheck-benchmark` artifact records exit codes, diagnostics, commands, source/config/lock hashes, exact dependencies, runner image and dirty state. Both tools must pass. Exclude benchmark-enabled runs from normal PR wall-time comparisons because they intentionally add work.
5. Observe external Codecov/reviewer completion for the actual PR trigger. Record unavailable trigger provenance as unknown. Accept the candidate only after all retained checks pass and GitHub runner samples demonstrate lower overall latency without introducing flaky tests or losing the distinct behavior guarantees documented above. Review any coverage change against the removed scenarios; do not require extra tests solely to retain the prior 91.42% measurement.

Example dispatch commands, after authorization and confirming availability (wait for each run to finish):

```bash
gh workflow run tests.yml --ref brian/ci-latency-investigation -f pytest-workers=0 -f benchmark-types=false
gh workflow run tests.yml --ref brian/ci-latency-investigation -f pytest-workers=2 -f benchmark-types=false
gh workflow run tests.yml --ref brian/ci-latency-investigation -f pytest-workers=4 -f benchmark-types=false
gh workflow run tests.yml --ref brian/ci-latency-investigation -f pytest-workers=4 -f benchmark-types=true
```

For a local reproducibility check, use `uv sync --locked --all-extras --group dev --python 3.12` followed by `uv run --no-sync python .github/scripts/benchmark-typecheck.py`. The result explicitly labels local execution and must not be cited as GitHub runner performance. Keep the generated `typecheck-benchmark/` evidence outside commits.
