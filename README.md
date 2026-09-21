# Agentic Autonomous Classification — V1

An autonomous agent framework that takes a tabular **classification** problem, figures out
how to solve it, builds and validates models, and submits the result to Kaggle — without a
human in the loop.

Reference / inspiration: [1st Place — Distributed Intelligence, NVIDIA Inference Hub (Playground S6E8)](https://www.kaggle.com/competitions/playground-series-s6e8/writeups/1st-place-distributed-intelligence-nvidia-infe)

Reference target for V1: [Playground Series S6E9 — Predicting Electric Vehicle Purchases](https://www.kaggle.com/competitions/playground-series-s6e9)

---

## 1. Core idea

The LLM is a **planner and code author**, never a predictor.

Language models propose: candidate feature engineering, candidate model configs, candidate
ensembling strategies. Classical gradient boosting and linear models do the actual
prediction. Every LLM proposal is then scored by real cross-validation on real data, and the
scoreboard — not the model's confidence — decides what survives.

This is what makes the loop safe to run unattended: a hallucinated feature either raises CV
score or it is discarded. There is no path by which a wrong LLM output silently becomes the
submission.

**Distributed intelligence** means several LLM backends work the same problem in parallel,
each producing an independent candidate branch. Diversity of proposers produces diversity of
solutions, which is exactly what ensembling wants.

---

## 2. Hard constraints

These are non-negotiable. Do not work around them.

| Constraint | Meaning |
|---|---|
| **No vendor SDKs** | No `openai`, `anthropic`, `google-genai`, `nvidia-*`, `langchain`, `llama-index`, `kaggle`. Every remote call is raw HTTP via `httpx`. |
| **Two LLM backends only** | A local server (Ollama / llama.cpp / vLLM) and the NVIDIA inference hub. Both are addressed through the same OpenAI-compatible `/v1/chat/completions` shape. |
| **No hosted-agent frameworks** | The orchestration loop is our own code. It must be readable and steppable in a debugger. |
| **Offline generated code** | Code the LLM writes runs with no network access. It reads local parquet/CSV and writes local artifacts. |
| **Deterministic training** | Fixed seeds, fixed fold assignment. Two runs on the same plan must produce the same CV score. |
| **Budget-capped** | Token spend, wall clock, and Kaggle submissions per day are all hard limits, enforced in code, not by convention. |

---

## 3. Architecture

```
                     ┌────────────────────────────────┐
                     │        Orchestrator            │
                     │  (state machine + run ledger)  │
                     └───────────────┬────────────────┘
                                     │
   ┌──────────┬──────────┬───────────┼───────────┬──────────┬──────────┐
   │          │          │           │           │          │          │
 Scout    Architect   Engineer    Trainer     Critic   Ensembler  Submitter
(profile) (plan)    (writes FE)  (no LLM)   (reflect)  (blend)   (Kaggle)
   │          │          │           │           │          │          │
   └──────────┴──────────┴───────────┴───────────┴──────────┴──────────┘
                                     │
                      ┌──────────────┴──────────────┐
                      │        LLM Router           │
                      │  local  │  NVIDIA hub       │
                      └─────────────────────────────┘
```

### Agent roster

*V1 roster. Section 17 replaces the Architect, Engineer, and Critic with the Researcher,
Analyst, Scholar, and Assessor; the Scout, Trainer (as baseline), Ensembler, Submitter, and
Historian keep their roles.*

Each agent is a class with one `run(ctx) -> AgentResult` method. Agents never call each
other; the orchestrator sequences them.

**Scout** — deterministic, no LLM. Loads train/test, produces a data profile: shapes, dtypes,
cardinality, missingness, target distribution and imbalance ratio, train/test distribution
drift per column, candidate ID column, candidate target column, memory footprint. Output is a
compact JSON profile — this is what every downstream prompt sees instead of raw data.

**Architect** — LLM. Reads the profile plus the competition metadata (metric, submission
format) and emits a structured `Plan`: preprocessing steps, feature engineering ideas, model
family shortlist with starting hyperparameters, CV scheme. Runs *N* times across different
backends to produce *N* divergent plans.

**Engineer** — LLM. Turns one plan's feature-engineering section into a single Python module
exposing `def build_features(train_df, test_df) -> (train_df, test_df, list[str])`. Nothing
else. It is validated by execution, not by reading.

**Trainer** — deterministic, no LLM. Given a plan and a feature module, runs stratified K-fold
for each model family, produces OOF predictions, test predictions, fold scores, timing, and
feature importance. This is the only component that touches the target.

**Critic** — LLM. Reads fold scores, OOF error analysis, and any traceback. Decides one of:
`refine` (adjust this branch), `abandon` (branch is a dead end), `promote` (good enough for
the ensemble pool). Must justify with numbers from the run, not vibes.

**Ensembler** — mostly deterministic. Takes all promoted branches' OOF matrices and searches
blend weights (hill climbing on OOF, plus a rank-average and a logistic stacker baseline).
Picks whichever wins on OOF. LLM only used to suggest which subsets to try when the pool is
large.

**Submitter** — deterministic. Writes `submission.csv` against the sample format, validates
row count / column names / ID alignment / value range, then uploads via the Kaggle HTTP API
and polls for the public score.

**Historian** — deterministic. Appends every branch (plan hash, features, params, CV, LB) to a
run ledger. On a new run against the same competition, the top entries are injected into the
Architect prompt as prior knowledge.

---

## 4. Repository layout

```
aac/
  __init__.py
  cli.py                  # entry point: aac run --config configs/s6e9.yaml
  orchestrator.py         # state machine, parallel branches, budget enforcement
  context.py              # RunContext: paths, config, ledger handle, budget counters
  config.py               # pydantic config, ${VAR} expansion, secrets masked
  env.py                  # secrets.yml loader, credential report by source
  doctor.py               # aac doctor
  baseline.py             # aac baseline: constant-prediction round trip
  plan.py                 # Plan schema (Architect output, Trainer input), default plan
  llm/
    client.py             # raw httpx client for OpenAI-compatible /v1/chat/completions
    router.py             # task tier -> backend/model selection, failover
    schema.py             # JSON contracts + validate/repair loop
    prompts/              # one .md per agent, versioned, no f-strings inline
  kaggle/
    api.py                # raw HTTP: metadata, download, submit, poll
    data.py               # runs/_data/{slug} cache: zip, extracted files, parquet
    submission.py         # format validation before upload, daily budget
  agents/
    scout.py architect.py engineer.py trainer.py
    critic.py ensembler.py submitter.py historian.py
  exec/
    sandbox.py            # subprocess runner: timeout, import allowlist, no network
    artifacts.py          # run dir layout, atomic writes
  models/
    registry.py           # lgbm / xgb / catboost / hist-gbdt / logistic wrappers
    prepare.py            # feature matrix: numeric float64, shared category vocabularies
    cv.py                 # fold generation, OOF assembly, per-fold target encoding
    metrics.py            # Kaggle metric name -> implementation, scoring
    target.py             # class order and positive label
  ledger.py               # SQLite: runs, branches, scores, submissions
configs/
  s6e9.yaml
  spaceship-titanic.yaml  # second competition: accuracy metric, True/False label submission
runs/                     # gitignored; one dir per run
tests/
```

### Dependencies

`httpx`, `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `xgboost`, `catboost`, `pyyaml`,
`pydantic`, `rich`, `optuna`. Nothing else without a reason written in the PR.

---

## 5. LLM layer

### 5.1 Client

One function, used by everything:

```python
def complete(
    messages: list[dict],
    *,
    backend: Backend,      # resolved by the router
    json_mode: bool = False,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    timeout: float = 120.0,
) -> Completion: ...
```

It POSTs to `{base_url}/chat/completions` with an `Authorization: Bearer` header when the
backend needs one. Retries on 429 / 5xx / timeout with exponential backoff and jitter, capped
at 4 attempts. Every call — request, response, token counts, latency, cost estimate — is
logged to the ledger.

### 5.2 Backends

```yaml
backends:
  local:
    base_url: ${LOCAL_LLM_BASE_URL:-http://localhost:11434/v1}
    model: qwen2.5-coder:14b
    api_key: null
    cost_per_1k: 0.0
  nvidia:
    base_url: ${NVIDIA_BASE_URL:-https://integrate.api.nvidia.com/v1}
    model: <pick a strong reasoning/coding model available on the hub>
    api_key: ${NVIDIA_API_KEY}
```

Confirm the exact model identifiers against the hub's model list endpoint at startup rather
than hardcoding them — availability changes. Fail loudly at startup if a configured model is
not in `GET /v1/models`.

### 5.3 Router

Route by task tier, not by agent name:

| Tier | Tasks | Default backend |
|---|---|---|
| `cheap` | profile summarisation, naming, JSON repair, classification of Critic verdicts | local |
| `code` | Engineer feature modules | local first, escalate to NVIDIA after 2 failed executions |
| `reason` | Architect plans, Critic analysis, ensemble strategy | NVIDIA |

Failover: if the primary backend errors out or returns unparseable JSON twice, fall through to
the other backend and record the escalation. If both fail, the branch is marked `failed` — the
run continues with the other branches.

For the distributed-intelligence effect, the Architect is invoked `n_branches` times with
different `(backend, model, temperature)` triples so plans genuinely diverge.

### 5.4 Structured output without an SDK

No tool-calling, no schema enforcement from a library. Do this instead:

1. Put the JSON schema in the prompt, with one filled example.
2. Send `response_format: {"type": "json_object"}` when the backend supports it; ignore the
   field otherwise.
3. Extract: try `json.loads` on the whole body, then on the first fenced block, then on the
   first balanced `{...}` span.
4. Validate against a pydantic model.
5. On failure, send one repair turn containing the raw output and the validation error, at
   `temperature=0`. One repair only.
6. Still failing → the caller gets `None` and decides. Never guess at the missing fields.

Every prompt lives in `llm/prompts/*.md` with a version header. Prompts are data, not code.

---

## 6. Kaggle layer

Auth: `KAGGLE_ACCESS_TOKEN` (a `KGAT_…` access token) sent as `Authorization: Bearer`. It is read
from the environment, from `secrets.yml`, or from `~/.kaggle/access_token` where the official CLI
stores it. Fallback: `KAGGLE_USERNAME` + `KAGGLE_KEY` with HTTP basic auth. Base URL
`https://www.kaggle.com/api/v1`. Verified 2026-09-05: bearer auth works on the metadata, file-list
and submission-list endpoints; unauthenticated calls return 401.

Endpoints, as verified on 2026-09-05 against the official client (kaggle 2.2.4) and the live
service. Every call is `POST https://api.kaggle.com/v1/competitions.CompetitionApiService/{Method}`
with a camelCase JSON body; fields at their default value are omitted from responses.

- `ListCompetitions {search}` — metadata including `evaluationMetric`. **Read the metric from
  here; never hardcode it.** The framework handles accuracy, AUC, log loss, F1, and macro-F1,
  and refuses to run on a metric it does not implement rather than silently substituting one.
- `ListDataFiles {competitionName}` — file inventory.
- `DownloadDataFiles {competitionName}` — 302 to a signed storage URL. Fetch it with a bare
  GET: no credentials and no `Content-Type` header, because the signature covers that header
  and a JSON content type carried over from the POST yields `SignatureDoesNotMatch`.
- `ListSubmissions {competitionName}` — history, paginated by `nextPageToken`; used to count
  today's submissions (UTC day) and to poll for a score after upload.
- Submission handshake: `StartSubmissionUpload {competitionName, fileName, contentLength,
  lastModifiedEpochSeconds}` returns `{token, createUrl}`; PUT the raw bytes to `createUrl`
  (200/201, retry on 503); then `CreateSubmission {competitionName, blobFileTokens,
  submissionDescription}` returns `{ref}`. Poll `ListSubmissions` until the ref is `COMPLETE`
  or `ERROR`. `aac baseline` submits a constant prediction and is the integration test.
- Never send cookies back: the gateway sets `ka_sessionid` on every response, and a request
  carrying it is treated as anonymous (401) even with a valid bearer token.

Cache downloaded data under `runs/_data/{slug}/`; never re-download inside a loop.

Submission budget: playground competitions allow a small number per day. Config key
`kaggle.max_submissions_per_day`, default 5, enforced by counting the submission list before
every upload. The framework submits only the current best ensemble, and only when its OOF
score beats the last submitted OOF by more than `min_improvement`.

---

## 7. Execution sandbox

LLM-authored code runs in a subprocess, never in the orchestrator process.

- Separate `python -I` subprocess (`aac/exec/_runner.py`), cwd set to the branch's sandbox
  dir, a minimal environment, fixed thread counts, `PYTHONHASHSEED=0`.
- Wall-clock timeout (default 900s) and a memory cap via `RLIMIT_AS` where the OS honours
  it (Linux; macOS ignores it). Killed processes mark the attempt `timeout` with the reason
  captured for the Engineer and the Critic.
- Import allowlist: `pandas`, `numpy`, `scipy`, `sklearn`, `itertools`, `math`, `re`,
  `collections`, `warnings`, plus the harmless `typing`, `functools`, `string`, `datetime`,
  `dataclasses`. Anything else is a rejection before execution.
- Static rejection (AST pass) of `open`, `eval`, `exec`, `__import__`, `compile`, `getattr`
  and friends, dunder escape hatches such as `__subclasses__` and `__globals__`, relative
  imports, and any function other than `build_features(train_df, test_df)`.
- Runtime shims inside the subprocess: `socket.socket` cannot be instantiated, and `open`
  refuses to write outside the sandbox directory.
- The module must be pure: the runner calls it twice on fresh copies and refuses any
  difference in the outputs. It also checks row count, row order, the id column, that the
  target column is absent, and that new columns hold scalars.
- Leakage tripwire: the inputs handed to the sandbox never contain the target column, and
  the static pass rejects any string or attribute equal to the target name. The Engineer
  prompt itself never names the target.

This is best-effort isolation, not a security boundary: macOS has no cheap network
namespaces. It is designed to catch an LLM's mistakes, not a hostile author.

## 8. State and resumability

SQLite ledger at `runs/ledger.db`:

- `runs(id, slug, started_at, config_hash, status)`
- `branches(id, run_id, plan_json, plan_hash, status, cv_mean, cv_std, fold_scores, duration)`
- `llm_calls(id, run_id, branch_id, agent, tier, backend, model, prompt_tokens, completion_tokens,
  latency, cost, ok, error, attempts, escalated)`
- `submissions(id, run_id, filename, oof_score, public_score, submitted_at)`

Every branch writes its artifacts to `runs/{run_id}/branches/{branch_id}/`: `plan.json`,
`features.py`, `oof.npy`, `test_pred.npy`, `metrics.json`, `stdout.log`.

`aac run --resume {run_id}` reconstructs state from the ledger and continues. A crashed run
must never lose completed branches.

---

## 9. The loop

*V1 loop. The V2 loop of section 17 keeps the outer stages and replaces the per-plan
iteration with Researchers on a live ensemble pool that uploads on improvement.*

```
load config
  -> fetch competition metadata + data          (Kaggle)
  -> profile                                    (Scout)
  -> retrieve prior knowledge for this slug     (Historian)
  -> generate N plans in parallel               (Architect x N backends)
  -> for each plan, up to max_iters:
        write feature module                    (Engineer)
        execute in sandbox                      (exec)
        train + CV                              (Trainer)
        judge                                   (Critic) -> refine | abandon | promote
  -> blend promoted branches                    (Ensembler)
  -> validate + submit best                     (Submitter)
  -> poll public score, write back to ledger    (Historian)
```

Branches run concurrently with a semaphore on both LLM calls and CPU-bound training. The run
stops on whichever comes first: `max_iters` exhausted, time budget, token budget, or no branch
improving OOF by `min_improvement` for `patience` iterations.

---

## 10. Validation rules

These protect the run from itself:

- **One fold assignment for the whole run.** Generated once from the seed, reused by every
  branch. Cross-branch score comparison is meaningless otherwise, and so is blending.
- **OOF is the only ranking signal.** The public leaderboard is recorded but never optimised
  against.
- **Leakage tripwire.** Feature modules receive train and test together only for fitting
  encoders on train statistics; any encoder must be fitted inside the fold loop. A branch whose
  CV beats the previous best by more than `leak_threshold` (default 0.05 absolute) is
  automatically re-run with shuffled targets — if the score stays high, it is a leak, and the
  branch is killed.
- **Degenerate check.** An experiment whose predictions are constant, or whose OOF beats a
  constant prediction at the class rate by less than `run.degenerate_margin` (default 0.002),
  is recorded as failed with kind `degenerate`, never as a success. The same check applies to
  Assessor interviews.
- **Submission validation.** Row count matches sample submission, ID column matches exactly and
  in order, no NaN, probabilities in `[0,1]`, class labels drawn from the training label set.
- **Determinism check.** In CI, the same plan run twice must produce identical CV to 1e-9.

---

## 11. Config

*V1 keys. V2 adds `competition.extra_train`, `competition.extra_flag`, `competition.notes`,
`researchers`,
`scholar` (with `reuse_runs`), `assessor`,
`run.max_rounds`, `run.max_consecutive_failures`, `run.share_every`, `run.degenerate_margin`,
`run.schedule`, `models.variants`, `models.variant_families`, `models.variant_max_seconds`,
`models.low_cardinality_max`,
`models.seed_bag`, `models.seed_bag_top`, `models.seed_bag_max_seconds`,
`backends.*.fallback_backend`, `backends.*.fallback_model`, `backends.*.trip_after`,
`backends.*.cooldown_seconds`, `sandbox.experiment_timeout_seconds`,
`sandbox.determinism_rows`, and drops `architect` and `run.n_branches`; see `configs/`.*

```yaml
competition:
  slug: playground-series-s6e9
  target: null           # null = Scout infers it
  id_col: null

run:
  seed: 42
  n_folds: 5
  n_branches: 4
  max_iters_per_branch: 3
  patience: 2
  min_improvement: 0.0005
  max_wall_clock_minutes: 180
  max_tokens_total: 2000000
  parallel_branches: 2

models:
  enabled: [lightgbm, xgboost, catboost, hist_gbdt, logistic]
  tuning: optuna
  tuning_trials: 30

kaggle:
  submit: true
  max_submissions_per_day: 5

backends: { ... }         # see §5.2
```

---

## 12. CLI

```
aac run       --config configs/s6e9.yaml [--dry-run] [--no-submit] [--poll-timeout S] [--repeat N]
aac resume    --run-id 20260905-1432 [--config ...] [--no-submit]   # continue from the ledger and disk
aac profile   --config configs/s6e9.yaml [--json]                  # Scout only, prints the profile
aac replay    --experiment-id <run>-<agent>-r<n>                   # re-execute a stored experiment, check 1e-9
aac submit    --config ... --run-id ...                            # upload a stored run's file (after quota reset)
aac baseline  --config configs/s6e9.yaml [--no-submit]             # constant-prediction round trip
aac ledger    --slug playground-series-s6e9                        # runs, branches, experiments, notes, submissions
aac scores    --slug playground-series-s6e9                        # write back public scores that were still pending
aac doctor    [--config ...]                                       # env vars, backends, JSON contract, Kaggle
```

`aac doctor` must pass before anything else is worth debugging: it pings both LLM backends
with a one-token completion, lists available models, and hits the Kaggle competitions endpoint.

---

## 13. Definition of done for V1

The framework is complete when a single command:

```bash
aac run --config configs/s6e9.yaml
```

does all of the following unattended:

1. Authenticates to Kaggle, pulls S6E9 metadata and data, reads the evaluation metric from the
   API.
2. Profiles the data and infers target and ID columns without them being configured.
3. Produces at least 3 divergent plans from at least 2 different LLM backends. *(V2, amended
   2026-09-08: at least 3 Researcher seats on distinct tracks; the second backend stays
   configured and verified as a fallback, a seat on it is optional after its track record.)*
4. Generates feature code that executes cleanly, or recovers from a traceback within the retry
   budget.
5. Trains at least 3 model families with a shared fold assignment and produces valid OOF.
6. Blends and beats every individual branch's OOF score.
7. Writes a submission that passes format validation.
8. Uploads it and records the returned public score in the ledger.
9. Stays inside the token, time, and submission budgets.
10. `aac ledger` afterwards shows a readable history of what was tried and what it scored.

A run that fails a stage must fail loudly, write the reason to the ledger, and leave the run
directory inspectable.

---

## 14. Build order

Do not build this breadth-first. Each milestone must be runnable before the next starts.
Progress, deviations, and findings are tracked in `BACKLOG.md`, updated after every milestone.
Items 6 to 10 below are superseded by the V2 build order in section 17.4; items 1 to 5 are done.

1. **Skeleton + doctor.** Config loading, run dirs, ledger schema, `aac doctor` green against
   both backends and Kaggle.
2. **Kaggle I/O.** Metadata, download, and a hand-written constant-baseline submission that
   comes back with a real public score. This de-risks the submission handshake early.
3. **Deterministic core, no LLM.** Scout + a hardcoded plan + Trainer + CV + Submitter. This is
   already a working autopilot and becomes the fallback path when every LLM branch fails.
4. **LLM client + router + JSON contract.** With the validate/repair loop and full call logging.
5. **Architect and Engineer.** Single branch, single iteration, sandbox execution.
6. **Critic and the iteration loop.** Refine / abandon / promote.
7. **Parallel branches across backends.** The distributed part.
8. **Ensembler.**
9. **Historian priors and resume.**
10. **Hardening.** Leakage tripwire, determinism test, budget enforcement tests, a full
    end-to-end run on S6E9.

---

## 15. Out of scope for V1

Regression and multi-label targets. Image, text, and time-series inputs. GPU training.
Distributed training across machines. A web UI. (Neural network families were out of scope
for V1 and are in scope from section 17 on.) Any Kaggle
competition type other than a simple file-upload submission. Note these as V2 candidates and
do not let them shape the V1 abstractions.

---

## 16. Environment

Secrets live in `secrets.yml` at the repo root (gitignored; copy `secrets.example.yml`). It is a
flat mapping of environment variable names to values and is exported into the process
environment at startup. Real environment variables always take precedence, so CI and shells can
override it without editing the file.

```yaml
NVIDIA_API_KEY: nvapi-...
KAGGLE_ACCESS_TOKEN: KGAT_...
# Optional overrides (defaults shown):
# NVIDIA_BASE_URL: https://integrate.api.nvidia.com/v1
# LOCAL_LLM_BASE_URL: http://localhost:11434/v1
```

`AAC_SECRETS_FILE` or `aac --secrets PATH` points at a different file. Keep the file mode `600`.

Never commit `secrets.yml`, `kaggle.json`, or any key. `aac doctor` reports which credentials are
present by source, never their values.

---

## 17. Distributed intelligence (V2 direction)

*Status 2026-09-06: milestones 7 to 12 of 17.4 are complete and verified with unattended runs
on both competitions; see `BACKLOG.md` for results, findings, and open risks.*

Decided 2026-09-06 after reading the reference write-up (`docs/reference-s6e8-writeup.md`).
The goal is the write-up's method with the two backends this project allows, a local server
and the NVIDIA hub, over raw HTTP and our own loop. The V1 core of sections 1 to 13 stays as
the harness and the fallback. Three V1 rules are amended:

- The LLM may write model code, not only feature code, but only inside the fold-safe harness
  of 17.1. It still never predicts by itself and never sees a validation fold's target.
- The sandbox allowlist gains the model libraries and `torch`. Neural models are in scope.
- Submissions happen during the run whenever the ensemble's honest OOF improves, not only at
  the end. The daily quota and `min_improvement` still gate every upload.

The V1 Architect and Engineer (section 3) were retired on 2026-09-06; the Researcher of 17.2
replaces both. The Scout, Trainer (as the baseline), Submitter, and Historian roles stand.

Kept unchanged: no vendor SDKs, no hosted agent frameworks, deterministic training, one fold
assignment per run, OOF as the only ranking signal, the 3-hour wall clock, token and
submission budgets, the deterministic default plan as baseline and fallback.

### 17.1 The experiment harness

An experiment is one Python module written by an agent:

```python
def build_features(train_df, test_df) -> (train_df, test_df, list[str])   # optional, target-free
def fit_predict(X_train, y_train, X_valid, X_test, meta) -> (p_valid, p_test)
```

The runner owns the folds. It calls `fit_predict` once per fold with that fold's training rows
and target only, assembles the OOF matrix, scores it with the competition metric, and writes
`oof.npy`, `test_pred.npy`, `metrics.json`. `meta` carries the class count, the categorical
column names, the seed, and the thread count. Anything target-dependent (target encoding,
stacking, early stopping) has to happen inside `fit_predict`, where only the training fold's
target exists. Leakage is impossible by construction, not by review. Determinism: two runs of
the same module must agree to 1e-9; the runner checks it on a subsample.

Allowed imports: the V1 list plus `lightgbm`, `xgboost`, `catboost`, `torch`, `optuna`.
Per-experiment wall-clock and memory limits come from config.

### 17.2 Agents

- **Scout**: unchanged.
- **Researcher** (LLM) replaces Architect and Engineer. One per `(backend, model, track)`.
  Each round: read the knowledge base (own history, the track leaderboard, the leader's best
  experiment, research notes), state a hypothesis, write the experiment module, run it, read
  the score and the Analyst's diagnostics, record. Rounds continue until the agent's share of
  the budget is spent or `patience` rounds pass without improvement.
- **Analyst** (deterministic, was Critic): fold spread, calibration, feature importances,
  worst slices, comparison with the leader. Its text goes into the Researcher's next prompt.
  Verdicts are rules, not LLM opinions: continue while improving, stop after `patience`,
  abandon after repeated failures.
- **Tracks**: `gbdt`, `linear`, `neural`, `open`. Agents on one track compete; every
  `share_every` rounds the trailing agent's prompt receives the leader's experiment and
  notes. This is the write-up's tip sharing. A leader module that imports a library the
  receiver's track forbids is shared as its hypothesis only, never as code.
- **Schedule**: seats play round-robin (`run.schedule`): every seat gets round n before any
  seat gets round n+1, so a slow or failing seat cannot starve the others of the wall clock.
  Each prompt carries the minutes left for the whole team.
- **Owner notes** (`competition.notes`): what the owner knows about the competition, a leak
  or a trap, written as a note every Researcher reads in every round.
- **Scholar** (LLM, research packets): the strongest slow hub model, asked for feature and
  modelling ideas only, no code. Stored as notes that every Researcher reads, with the ideas
  about libraries outside the reader's track stripped at prompt time. Packets describe the
  competition, not the run, so the latest ones are reused for `scholar.reuse_runs` later
  runs before the Scholar is asked again.
- **Assessor** (interviews): a fixed small task per configured model, scored by CV. The
  resulting track record (valid modules, runs that succeeded, best OOF, tokens, latency)
  drives the allocation of tracks and rounds in later runs.
- **Original data** (`competition.extra_train`): the Kaggle dataset a Playground
  competition was generated from is downloaded once into `runs/_data/datasets/`, aligned
  with the train columns (extra columns dropped, dtypes coerced, rows identical to a
  synthetic row removed, fresh negative ids) and appended to the training part of every fold,
  never to validation or test, so every OOF stays a score on the competition's own rows. A
  flag feature (`competition.extra_flag`, default `is_original`) is 1 on those rows and 0
  elsewhere. Deterministic branches, seed bags and Researcher modules all see them; a module
  finds them as the last `meta["n_extra"]` rows of `X_train`.
- **Deterministic branches**: the default plan on every enabled family (`lightgbm`,
  `lightgbm_focal`, `xgboost`, `catboost`, `hist_gbdt`, `logistic`; the focal-loss LightGBM
  is a second loss that ranks rows differently, binary targets only), then the plan variants
  in `models.variants` on every family that trained within `models.variant_max_seconds` on
  the default plan, or on `models.variant_families` when set: `categorical` treats integer columns
  with at most `models.low_cardinality_max` distinct values as categorical levels, `encoded`
  target-encodes the categoricals and those integers inside each fold (the integers keep
  their numeric column too), `digits` goes after generator artefacts: low-order digits and
  moduli of every fine-grained numeric column, the frequency of each exact value over train,
  test and extra rows, and fold-internal exact-value target encoding of every raw column
  (`aac/models/features.py`). Then seed bags: the best `models.seed_bag_top` tree families
  are retrained with `models.seed_bag` extra seeds. Every family of every branch is a pool
  member. No LLM is involved, so these levers are tested on every run at a fixed cost.
- **Ensembler**: the pool holds every experiment's OOF and test predictions across rounds
  and agents, plus every deterministic branch member. Hill climbing with replacement, rank
  average, and a logistic stacker on the shared folds; the best honest OOF wins.
- **Submitter**: uploads whenever the pool blend beats the last upload by `min_improvement`,
  inside the daily quota. An upload that Kaggle is still scoring when `--poll-timeout`
  passes is recorded without a score and never fails the run; the next run, `aac submit`, or
  `aac scores` writes the score back (and marks a run that failed only on that poll as
  completed). `--repeat N` starts N runs one after another; a failed run does not stop the
  next.
- **Historian**: the knowledge base across runs (experiments, notes, track records). Before
  the Researchers start it writes one `prior` note per track (the best earlier experiments a
  Researcher on that track may build on, judged on the module's imports, with the top
  module's code) and `pitfalls` notes: the distinct failures seen on the competition with
  counts (API slips, timeouts, leaks, degenerate output), plus each track's own import
  violations. Every Researcher reads them every round.

- **Router**: a failed call (after its retries) falls over to `backends.<name>.fallback_backend`
  and `fallback_model`, which may be another model on the same hub; without them, to the
  other backend. A circuit breaker skips a backend for `cooldown_seconds` after `trip_after`
  consecutive failed calls, so a dead hub costs one call, not four attempts per call.

### 17.3 Knowledge base

Ledger tables: `experiments(id, run_id, agent, backend, model, track, round, code_hash,
hypothesis, cv_mean, cv_std, oof_score, duration, status, error, parent)`,
`notes(id, run_id, source, kind, text, created_at, track)` (a note with a `track` reaches
only that track's Researchers), `model_track_record(backend, model, task, valid_modules,
runs_ok, best_oof, tokens, latency, updated_at)`. Experiment `kind` values: `ok`,
`no-code`, `track`, `rejected`, `error`, `timeout`, `leak`, `degenerate`.
Artifacts: `runs/{run_id}/experiments/{agent}/r{round}/` with `experiment.py`, `oof.npy`,
`test_pred.npy`, `metrics.json`, `analysis.json`, `stdout.log`. Datasets used as extra
training data are cached under `runs/_data/datasets/{owner}__{slug}/`.

### 17.4 Build order, in the article's order (replaces items 6 to 10 of section 14)

The reference write-up built its system in four phases; the milestones follow them.

7. **Harness and one Researcher.** The fold-owning experiment harness and a single agent
   writing `fit_predict` experiments with CV feedback each round. (In progress.)
8. **Phase 1: one autonomous agent builds the ensemble and submits by itself.** The
   Ensembler over every experiment (hill climbing with replacement, rank average, logistic
   stacker on the shared folds), the Researcher's objective changed from "best single
   score" to "raise the ensemble's honest score", diversity encouraged (families, seeds,
   features), and a submission whenever the blend improves by `min_improvement` inside the
   daily quota. The Kaggle competition description goes into the prompt. Retire the V1
   Architect and Engineer path.
9. **Phase 2: several agents, a shared knowledge base, single-model battles with tip
   sharing.** Researchers on different backends run in parallel on tracks (`gbdt`,
   `linear`, `neural`, `open`), read each other's experiments in the ledger, receive the
   Analyst's diagnostics, and every `share_every` rounds the trailing agent gets the leader's
   experiment. Rule-based continue / stop / abandon and per-agent budget shares.
10. **Phase 3: research packets.** The Scholar asks the strongest slow hub model, several
    prompts in parallel, for feature and modelling ideas only; the packets are notes every
    Researcher reads and is asked to test.
11. **Phase 4: interview the hub's models and assign jobs.** The Assessor runs a fixed small
    task per configured model, records a track record (valid modules, runs that succeeded,
    best OOF, ensemble gain, tokens, latency), and allocates tracks and rounds from it.
    Historian priors across runs and `aac resume`.
12. **Hardening and unattended runs**: leakage tripwire with the shuffled-target re-run,
    determinism and budget tests, full runs on both configs against section 13.
