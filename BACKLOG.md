# Backlog

Milestone plan for V1, following README section 14. **Update this file at the end of every
milestone**: mark it done with the date, list what was built and tested, record deviations from
the README, and note findings that change later work.

## Status

| # | Milestone | Status | Date |
|---|---|---|---|
| 0 | Secrets hygiene | done | 2026-09-05 |
| 1 | Packaging and CLI skeleton | done | 2026-09-05 |
| 2 | Config, ledger, run context, `aac doctor` | done | 2026-09-05 |
| 3 | Kaggle I/O and constant-baseline submission | done | 2026-09-05 |
| 4 | Deterministic core with no LLM (Scout, Trainer, CV, Submitter) | done | 2026-09-05 |
| 5 | LLM client retries, router, JSON contract, call logging | done | 2026-09-05 |
| 6 | Architect and Engineer, single branch, sandbox | done | 2026-09-06 |
| 7 | Harness and one Researcher | done | 2026-09-06 |
| 8 | Phase 1: Ensembler, auto-submit on improvement, retire V1 agents | done | 2026-09-06 |
| 9 | Phase 2: parallel agents, shared knowledge, tracks, tip sharing | done | 2026-09-06 |
| 10 | Phase 3: Scholar research packets | done | 2026-09-06 |
| 11 | Phase 4: Assessor interviews, track record, Historian, resume | done | 2026-09-06 |
| 12 | Hardening and unattended runs | done | 2026-09-06 |
| 13 | Track-safe knowledge, pitfalls memory, degenerate check | done | 2026-09-08 |
| 14 | Deterministic levers, hub resilience, round-robin seats, packet reuse | done | 2026-09-08 |
| 15 | Pending public scores, score backfill, `aac scores`, `--repeat` | done | 2026-09-09 |
| 16 | Original dataset as extra training rows | done | 2026-09-09 |

## Done

### 0. Secrets hygiene (2026-09-05)
- `secrets.yml` is YAML, gitignored, mode 600; `secrets.example.yml` is the committed template.
- `aac/env.py` exports the file into the environment at startup. Real env vars win. The Kaggle
  token also falls back to `~/.kaggle/access_token`. Doctor reports presence by source only.
- README sections 6 and 16 updated: Kaggle auth is a `KGAT_` bearer token, verified against
  the live API.

### 1. Packaging and CLI skeleton (2026-09-05)
- `pyproject.toml` with `uv.lock`, the `aac` console script, `ruff`, `pytest`.
- `aac/cli.py`: all six README subcommands parse with their required flags.
- `pyarrow` added beyond the README list for parquet I/O.

### 2. Config, ledger, run context, doctor (2026-09-05)
- `aac/config.py`: pydantic models for README section 11, `${VAR}` / `${VAR:-default}`
  expansion on string leaves, strict and non-strict loading, secrets as `SecretStr`, a
  config fingerprint that excludes secrets.
- `aac/ledger.py`: SQLite schema for runs, branches, llm_calls, submissions; WAL; thread-safe.
- `aac/exec/artifacts.py`: run directory layout, run id allocation, atomic writes.
- `aac/context.py`: `RunContext.create()` and the `Budget` counters (tokens, wall clock).
- `aac/models/metrics.py`: Kaggle metric display name to implementation mapping, scoring for
  auc, accuracy, logloss, f1, macro_f1. Unknown names are refused.
- `aac/llm/client.py`: single-attempt `complete()` and `list_models()` over raw httpx.
- `aac/kaggle/api.py`: bearer or basic auth, competition metadata.
- `aac/doctor.py`: credentials, config, ledger, every configured model on every backend
  (list plus a real completion, run concurrently), Kaggle metadata and metric resolution.
- `aac ledger --slug` lists runs and best branches.
- Tests: 88 passing, all offline (httpx MockTransport for every HTTP path). Live `aac doctor`
  is green against Ollama, the NVIDIA hub, and Kaggle.

**Findings from the live doctor run (2026-09-05):**
- The hub's `GET /v1/models` list is not proof a model is callable: `moonshotai/kimi-k2.6`
  is listed but returns 404 for this account. Doctor now completes on every configured model.
- Hub latency on a one-word prompt (max_tokens=64): `nemotron-3-super-120b-a12b` 1s,
  `openai/gpt-oss-20b` 1s, `minimaxai/minimax-m3` 0.5s, `nemotron-3-ultra-550b-a55b` 15s,
  `kimi-k3` 107s. `deepseek-v4-pro`, `deepseek-v4-flash`, `gemma-4-31b-it` and
  `mistral-nemotron` timed out at 200s or more. `mistral-large-2-instruct` and
  `llama-3.1-nemotron-ultra-253b` are listed but 404 like `kimi-k2.6`.
  Default `reason` model is `nvidia/nemotron-3-super-120b-a12b`; Architect branches use
  nemotron-3-super, gpt-oss-20b, minimax-m3, and the local qwen2.5-coder.
- Thinking models return `reasoning_content` and can spend the whole `max_tokens` budget on it,
  leaving `content` empty. Never ping with tiny `max_tokens`; JSON extraction (M5) must read
  `content` only.
- Concurrent requests to one hub model can hit HTTP 503 "worker request limit". Retries with
  backoff in M5 are required, not optional.
- S6E9: metric `Roc Auc Score`, deadline 2026-09-30, 10 submissions per day, files train.csv
  44.7 MB, test.csv 18.3 MB.

**Deviations from the README:**
- Kaggle auth is bearer token first, basic auth fallback (README said basic).
- Added config keys: `competition.metric` (explicit override only), `sandbox.*`,
  `run.leak_threshold`, `backends.*.timeout|supports_json_mode|max_concurrency`,
  `router.*`, `architect.branches`.
- `models/metrics.py` holds metric dispatch instead of `models/cv.py`; `cv.py` (M4) will
  import from it.

### 3. Kaggle I/O and constant-baseline submission (2026-09-05)
- `aac/kaggle/api.py` rewritten on the current API: JSON POSTs to
  `api.kaggle.com/v1/competitions.CompetitionApiService/{Method}`, bearer or basic auth,
  pagination, streamed download with size check, the three-step submission handshake with 503
  retry on the upload, and score polling.
- `aac/kaggle/data.py`: cache under `runs/_data/{slug}/` (zip, extracted CSVs, parquet), safe
  extraction, downloaded once and never inside a loop.
- `aac/kaggle/submission.py`: build, validate (columns, rows, id order, NaN, [0, 1], label set),
  atomic CSV write, UTC daily submission count and remaining budget.
- `aac/models/target.py`: class order and positive label inference shared by baseline, Scout,
  and Trainer.
- `aac/baseline.py` and `aac baseline --config ... [--no-submit]`: download, constant
  prediction, validation, upload, poll, ledger. Live run: submission 56033021 on S6E9 scored
  0.50000 public AUC, recorded in `runs/ledger.db`.
- `tests/integration/test_live_kaggle.py` repeats that round trip when `AAC_INTEGRATION=1`.
- Tests: 122 passing offline, with `tests/kaggle_fake.py` standing in for api.kaggle.com and
  the storage host.

**Findings (all now enforced by the fake server in tests):**
- The wire format was verified by reading kaggle 2.2.4 / kagglesdk 0.1.37 source in the
  scratchpad, not by importing it. The legacy `www.kaggle.com/api/v1` GET endpoints still work
  for metadata but the SDK no longer uses them; we follow the SDK.
- `ka_sessionid` cookie trap: the gateway sets it on every response; sending it back yields
  401 even with a valid bearer token. The client clears its cookie jar before every request.
- Redirect trap: httpx keeps `Content-Type: application/json` when turning the 302'd POST into a
  GET, and the storage signature covers that header, so the download fails with
  `SignatureDoesNotMatch`. The client follows the redirect manually with a bare GET.
- The `{"error": {"code", "message", "status"}}` envelope is the gateway's error shape;
  `{"code", "message"}` is the legacy one. Both are parsed.
- S6E9 data: train 668,665 rows x 15 columns, test 286,571 rows, bundle 14.8 MB. Target
  `Will_Buy_EV` is a `Yes`/`No` string, positive rate 0.174645, which is exactly the constant in
  the sample submission. Six string columns, eight numeric. The pandas 3 default string dtype
  is `str`.

**Deviations from the README:**
- Added `aac baseline` to the CLI (the README's "hand-written constant-baseline submission"
  made permanent) and `aac/kaggle/data.py` plus `aac/models/target.py` to the layout.
- README section 6 endpoint list replaced with the verified modern API.

### 4. Deterministic core with no LLM (2026-09-05)
Built against two competitions from the start, with no code change between them:
S6E9 (ROC AUC, probability submission, Yes/No target) and Spaceship Titanic (accuracy,
True/False label submission, booleans with missing values, composite id, free-text columns).

- `aac/agents/scout.py`: id and target inference with config overrides, target kind and
  positive label, submission shape (`proba`, `label`, `proba_per_class`), column typing
  (numeric, categorical, boolean, datetime, text, constant), missingness, numeric KS and
  categorical total-variation drift, warnings. Refuses regression-like targets, too many
  classes, per-class column mismatches, and sample/test row mismatches.
- `aac/plan.py`: the Plan schema the Architect will emit and the Trainer executes, with
  library defaults and a hash. `default_plan()` is the no-LLM fallback path.
- `aac/models/prepare.py`: target-free feature matrix, shared category vocabularies.
- `aac/models/registry.py`: lightgbm, xgboost, catboost, hist_gbdt, logistic wrappers with
  explicit seeds and thread counts; early stopping only on a seeded 10% carve-out of the
  training fold, never on the OOF fold; scikit-learn threads pinned with threadpoolctl.
- `aac/models/cv.py`: one stratified fold assignment per run saved to `folds.npy`, OOF and
  test assembly, per-fold target encoding.
- `aac/agents/trainer.py`: runs every family in the plan, tolerates a failing family, writes
  `oof_*.npy`, `test_*.npy`, `metrics.json` with importances and best iterations.
- `aac/agents/submitter.py`: predictions to the sample's shape (probability, label with the
  sample's dtype, or one column per class), validation, and the shared upload/poll/ledger
  path also used by `aac baseline`.
- `aac/orchestrator.py`: data, profile, folds, default plan as one branch, train, submit.
  `aac run --config ... [--dry-run] [--no-submit] [--poll-timeout]` and `aac profile
  --config ... [--json]` work.
- Config: `competition.files` overrides the filename heuristic (paths or globs, CSV or
  parquet); `run.n_jobs` and `run.max_classes` added.
- Tests: 166 passing offline, including the README section 10 determinism check for every
  family (identical OOF to 1e-9), Scout inference and refusals, per-fold target encoding,
  and end-to-end orchestrator runs against the fake Kaggle server.

**Live results (default plan, 5 folds, seed 42):**

| Competition | Best family | OOF | Public LB | Train time |
|---|---|---|---|---|
| spaceship-titanic | xgboost | 0.79650 accuracy | 0.79284 (submission 56033446) | 17 s |
| playground-series-s6e9 | xgboost | 0.94180 AUC | 0.94162 (submission 56033681) | 481 s |

S6E9 per family: xgboost 0.94180 (17 s), catboost 0.94166 (398 s), lightgbm 0.94164 (24 s),
hist_gbdt 0.94148 (33 s), logistic 0.93809 (9 s). Two separate runs produced identical
OOF scores for every family, so determinism holds on the real 668k-row table. CatBoost is 20x slower than XGBoost for the same score;
the Architect should be told this when it proposes model lists.

**Findings:**
- LightGBM 4.7 deprecates `eval_set` for `eval_X`/`eval_y`, and `eval_X` takes one matrix
  or a tuple, never a list. Naming categorical columns alongside it fails; `"auto"` from the
  pandas category dtype is the reliable path.
- scikit-learn's HistGradientBoosting has no `n_jobs`; without threadpoolctl it took every
  core and ran 10x slower than LightGBM on 8.7k rows.
- Ledger branch ids are global (`aac replay --branch-id`), so they are `{run_id}-{name}`;
  the directory name stays short.
- pandas 3 reads Yes/No and True/False columns as `str`/`object`; the Scout treats two-valued
  true/false strings as boolean and everything else as categorical.

**Deviations from the README:**
- `aac/models/target.py`, `models/prepare.py`, `plan.py` added to the layout (README
  section 4 updated).
- Early stopping is on an inner carve-out, not on the validation fold, so OOF stays honest.

### 6. Architect and Engineer, single branch, sandbox (2026-09-06)
- `aac/exec/sandbox.py` and `aac/exec/_runner.py`: AST allowlist and banned-call pass, target
  tripwire, `python -I` subprocess with socket shim, write guard, memory cap where honoured,
  purity check (module called twice), contract checks on rows, ids, columns, dtypes.
- `aac/agents/architect.py` with `prompts/architect.md`: Plan via the JSON contract with
  profile-aware validation (unknown columns, disabled families, chained feature references).
- `aac/agents/engineer.py` with `prompts/engineer.md`, `engineer_fix.md`, `code_repair.md`:
  fenced module, traceback fed back, escalation to the reason backend after
  `run.escalate_code_after` failures.
- Orchestrator: default branch plus one LLM branch per Architect spec; best OOF submitted;
  `verify_models()` fails loudly at startup; branch failures recorded, run continues.
- Tests: 246 passing (sandbox rejections, contract violations, purity, network and write
  shims, timeouts, Architect validation and repair, Engineer fix turns and escalation,
  end-to-end LLM branches, fallback to the default).

**Live results (four LLM branches each):**

| Competition | Default OOF | Best LLM branch OOF | Submitted | Public |
|---|---|---|---|---|
| spaceship-titanic | 0.79650 accuracy | 0.79604 (local qwen, catboost) | default | 0.79284 |
| playground-series-s6e9 | 0.94180 AUC | 0.94174 (nemotron plan, xgboost) | default | 0.94162 |

Every hub model produced a plan that flowed through the local Qwen Engineer into a trained
branch on S6E9. None beat the default plan: plain feature engineering on top of native
categorical handling does not move these tables, which is the argument for V2's model-level
experimentation. Findings fixed during the milestone: chained feature references were
rejected by the validator; engineered ratios produced `inf` (now mapped to NaN in
`prepare_matrix`); boolean-with-missing columns were cast with `astype(int)` three times in a
row (prompt now describes them and the fix turn asks for a change of approach).
Hub latency: minimax-m3 took 22 minutes for one branch on S6E9.

## Next

### 5. LLM client retries, router, JSON contract, call logging (2026-09-05)
- `aac/llm/client.py`: `complete()` retries on 429, 5xx, and transport errors with
  exponential backoff (1 s, doubled, capped at 30 s, up to 50% jitter), four attempts.
  Other 4xx and malformed bodies raise at once. `complete_once()` is the single attempt.
- `aac/llm/router.py`: tier to backend routing (`cheap`/`code` local, `reason` NVIDIA),
  explicit backend/model override, one failover hop to the other backend after the retries
  are spent, a per-backend `BoundedSemaphore` from `max_concurrency`, `verify_models()`
  against `GET /models`. Every call, failed or not, is a row in `llm_calls` with attempts and
  an `escalated` flag, and every successful call is charged to the run budget, which is
  checked before each call.
- `aac/llm/schema.py`: `extract_json` (whole body, fenced block, first balanced span, string
  aware), pydantic validation with readable errors, `describe_schema` for prompts, and
  `structured_completion`: one repair turn at temperature 0 carrying the raw reply and the
  error, then one hop to the other backend, then `None`. Nothing is ever guessed.
- `aac/llm/prompts.py` and `aac/llm/prompts/*.md`: front matter version, `## system` and
  `## user` sections, `{{placeholder}}` substitution that refuses to render with a gap.
  Shipped: `repair.md`, `ping.md`.
- `aac/ledger.py`: schema v2 with an in-place migration mechanism (`llm_calls.attempts`,
  `llm_calls.escalated`); a newer ledger is refused.
- `aac doctor` now runs the full JSON contract on each backend's default model.
- Tests: 207 passing offline with a scripted fake LLM server (`tests/llm_fake.py`) covering
  backoff timing, non-retryable errors, repair, failover, budget enforcement, concurrency
  limits, prompt rendering, and the v1 to v2 ledger migration.

**Live (doctor, 2026-09-05):** local qwen2.5-coder and nvidia nemotron-3-super both return
valid JSON first try. A real NVIDIA `ReadTimeout` was retried and succeeded. `minimax-m3`
answered in 6 s in one run and 114 s in the next: hub latency is volatile, and the wall
clock budget must be checked between every LLM call, which the router now does.

**Deviations from the README:**
- Failover is one hop, primary then the other backend, both for HTTP failures and for two
  unusable JSON replies; the README's "escalate `code` after 2 failed executions" is a
  Critic/Engineer policy and lands in M6/M7 on top of `Router.complete(backend=...)`.
- Ledger schema is now v2; README section 8 updated.

### Scope decision (2026-09-06)
The owner wants the write-up's method built on the local server and the NVIDIA hub. Decided:
agents write model code inside a fold-safe harness (`fit_predict` per fold, the runner owns
the folds), `torch` is allowed in the sandbox, the 3-hour wall clock stays, the deterministic
default plan stays as baseline and fallback. README section 17 is the spec; milestones 7 to 11
below replace the earlier plan.

### Roadmap reworked to the article's phases (2026-09-06)
The write-up built its system as: (1) one autonomous agent building a big diverse ensemble
and submitting by itself, (2) more agents plus a shared knowledge database, competing on
single models with tip sharing, (3) research packets, (4) interviewing hub models and
assigning jobs. Milestones 8 to 11 now follow that order; the ensemble and auto-submission
move ahead of the multi-agent battle. README section 17.4 is the spec.

### 7. Harness and one Researcher (done 2026-09-06)
- Done: fold-owning harness (`fit_predict` per fold, purity and determinism checks, torch
  allowed), Researcher loop with error feedback and patience, `experiments` and `notes`
  ledger tables, `researchers` config, candidate pool feeding the submission, budget
  exhaustion stops new work but still submits. 261 tests.
- Live, Spaceship (run 20260906-1114, 18 min): `nemotron-3-ultra` on the open track parsed
  Cabin into deck/num/side and Name into title/family size, fed CatBoost, and improved over
  four rounds to OOF 0.80973 against the deterministic 0.79650; public LB 0.80360 (was
  0.79284). Exit criterion met. `gpt-oss` ignored its neural track and wrote CatBoost;
  `minimax-m3` timed out five minutes per attempt and was dropped for nemotron-3-ultra; hub
  timeout lowered to 180 s.
- Live, S6E9 (run 20260906-1114-2, 25 min): default xgboost 0.94180 still best;
  `nemotron-3-ultra` 0.94145 (3/3 ok), local qwen linear 0.93809 (3/3 ok); the first two
  researchers failed all rounds. Submitted the default again, public 0.94162.
- S6E9 finding: the first two researchers failed all rounds on LightGBM 4.7 API changes
  (`early_stopping_rounds`, `verbose` in `fit()`), so the prompt now carries an environment
  card with installed versions and calling conventions, and `run.max_consecutive_failures`
  separates fixable errors from improvement patience.
- S6E9 rerun with the environment card (run 20260906-1139, 47 min): 10/14 experiments ran
  (6/12 before); nemotron-super 0.94153, gpt-oss 0.94151, nemotron-ultra 0.94134, local
  linear 0.93810; the deterministic xgboost 0.94180 still won and was submitted (0.94162).
  Single models plateau here; the ensemble (M8) is the lever on this table.
- Exit met: on Spaceship a Researcher beat the deterministic plan (0.80973 vs 0.79650, public
  0.80360 vs 0.79284); every experiment reproduces to 1e-9 (harness check, torch included).

### 8. Phase 1: one agent builds the ensemble and submits by itself
Rework steps, in order:
1. (done 2026-09-06) Retire the V1 Architect and Engineer path: remove the orchestrator branch loop,
   `architect.py`, `engineer.py`, their prompts and tests; move the reusable helpers
   (`extract_code`, `render_columns`, `size_hint`) next to the Researcher; drop
   `architect.branches` and `run.n_branches` from config.
2. (done 2026-09-06) Ensembler: pool = every ok experiment plus the deterministic baseline's families;
   hill climbing with replacement (Caruana), rank average, logistic stacker fitted on the
   shared folds; the best honest OOF wins; `ensemble.json` and blended `test_pred.npy`.
3. (done 2026-09-06) Researcher objective: the prompt shows the current ensemble score and each experiment's
   marginal gain to the blend; success is "the ensemble improved", diversity is rewarded
   (families, seeds, feature sets); patience counts ensemble gains, not single scores.
4. (done 2026-09-06) Submit on improvement: after every experiment, re-blend; upload when the blend beats
   the last upload by `min_improvement` inside the daily quota; ledger row per upload.
5. (done 2026-09-06) Competition description in the prompt (from the Kaggle metadata), plus `aac ledger`
   showing ensemble history.
Exit: on both configs the blend beats every single experiment and uploads happen during the
run without a human.
Live M8 result (Spaceship, run 20260906-1200, 32 min): 13-member pool; uploads on improvement
at 0.79565, 0.80266, and 0.80360 public; the best experiment reached OOF 0.81399 (ultra, open
track) but the config's daily cap of 5 was spent and the final upload raised, failing the run
at the last step. Fixed: the final upload now skips gracefully under the quota, the cap is 8,
and `aac submit --run-id` uploads a stored run's file later. `gpt-oss` on the neural track:
0/4 experiments in 17 minutes; M9's track rule forces torch, to be judged by the Assessor.
Live findings (Spaceship, 2026-09-06): the first M8 run showed the pool score dropping when a
member joined (hill climbing restarted from the best single each time) and an upload firing on
that worse blend (the upload rule only looked at the last upload). Fixed the same day: the
previous blend is a candidate and warm-starts the climb, so the pool is monotone, and uploads
require a positive gain. Also seen: a gbdt-track researcher writing a neural network, which M9
enforces statically.

### 9. Phase 2: several agents, shared knowledge, single-model battles
- Researchers in parallel (threads, LLM and CPU semaphores, per-agent budget shares).
- Shared knowledge base: each prompt carries the leaderboard of all agents' experiments.
- Tracks: `gbdt`, `linear`, `neural`, `open`; Analyst diagnostics; leader-to-trailer
  sharing every `share_every` rounds; rule-based continue / stop / abandon.
Exit: four agents on S6E9 inside the budgets with one documented case of a trailing agent
improving after receiving the leader's experiment.
Code done 2026-09-06: `ThreadPoolExecutor` over `run.parallel_branches` workers with the
router's per-backend semaphores; `LivePool` locked; `TeamState` holds each agent's best and
renders the team leaderboard plus the leader's module as a tip every `share_every` rounds to
a trailing agent; `check_track` rejects modules outside their track before execution (kind
`track`, fed back); `aac/agents/analyst.py` writes fold spread, calibration, weakest
categorical slices, correlation with the blend, and distance to the leader into the next
prompt and `analysis.json`. Per-agent budget shares are not implemented: the run-level
budget check before every round is the guard, and each agent's `rounds` caps its share.

### 10. Phase 3: research packets
- Scholar prompt on the strongest slow hub model, several parallel prompts, ideas only,
  stored as notes; Researchers are asked to test them.
- Done 2026-09-06: `aac/agents/scholar.py` with `prompts/scholar.md`; three default angles in
  parallel through the JSON contract (`Packet` of `Idea`s validated against the profile's
  columns); packets stored as `research` notes that `render_notes` puts into every
  Researcher prompt. Config `scholar:` (nemotron-3-ultra in both YAMLs).

### 11. Phase 4: interviews, track record, Historian, resume
- Assessor task per configured model; `model_track_record`; allocation of tracks and
  rounds; priors across runs; `aac resume`.
- Done 2026-09-06: `aac/agents/assessor.py` interviews each configured model on a stratified
  subsample of the real competition (one experiment through the harness), records
  `model_track_record` (ledger v4), re-interviews models with no working module, and
  `allocate()` drops a model after `assessor.skip_after_failures` failed interviews.
  `aac/agents/historian.py` writes `prior` notes from the best experiments of earlier runs
  including the top module's code. `aac resume --run-id` rebuilds the default branch and
  every agent's history from the ledger and disk, continues the remaining rounds, and
  re-blends; `aac submit --run-id` uploads a stored run's file after a quota reset.

### 12. Hardening
- Leakage tripwire with the shuffled-target re-run; determinism, budget, resume tests;
  full unattended runs on both configs against README section 13.
- Done 2026-09-06: `leak_check` in the Researcher loop re-runs any experiment that jumps more
  than `run.leak_threshold` over the best known score with shuffled targets and rejects it
  (kind `leak`) when it still beats chance by half the threshold; `aac replay
  --experiment-id` re-executes a stored module and reports the max absolute OOF difference
  against the stored array (README section 10's 1e-9). Determinism, budget, quota, resume, and
  replay are covered by tests (274 passing). Remaining: full unattended runs on both configs
  against README section 13, and the README section 13 checklist itself.

## Live results, 2026-09-06 afternoon (M8 to M11 code)
- S6E9 run 20260906-1231 (interviews, 3 Scholar packets, parallel researchers): stopped by
  hand when the local model stalled under memory pressure, then finished with
  `aac resume`, which reloaded 5 default families and 14 experiments and continued the
  local researcher's last round. Pool of 10: stacker 0.94202, hill climb 0.94196, best
  single 0.94180 (default xgboost); the blend choice wrongly applied `min_improvement` as
  a tie-breaker and submitted the single (0.94162 public). Fixed: blends compete on the
  honest OOF alone; S6E9 `min_improvement` lowered to 0.0001.
  Re-resumed after the fix: `stack_logistic` of 10 members OOF 0.94202 uploaded, public
  0.94178 (single model 0.94162). The blend beats every single model on S6E9 on both OOF
  and the leaderboard.
- Spaceship run 20260906-1200: 13-member pool, best single 0.81399, uploads at 0.79565,
  0.80266, 0.80360 (public best of the day); the stored 0.81399 blend uploaded later by
  `aac submit` scored 0.80009 public.
- Researchers on S6E9: nemotron-super 3/4 ok (best 0.94170), nemotron-ultra 1/4, gpt-oss 0/4
  on the neural track (retired), local qwen 1/4 on linear (target-name rejection, two track
  violations, then a working logistic at 0.93809).

## README section 13, definition of done: status 2026-09-06
1. Kaggle auth, metadata, metric from the API: done (bearer token, `Roc Auc Score` -> auc).
2. Profile infers target and id without config: done on both competitions.
3. At least 3 divergent plans from 2 backends: done as experiments from nemotron-super,
   nemotron-ultra, and the local qwen (three models, two backends).
4. Generated code executes or recovers within the retry budget: done (fix turns, track
   rule, environment card, failure allowance).
5. At least 3 families on shared folds with valid OOF: done (five in the default branch,
   more in experiments, one `folds.npy` per run).
6. Blend beats every individual OOF: done on S6E9 (0.94202 vs 0.94180) after the blend
   choice fix; on Spaceship the accuracy blend tied the best single (thresholded metric).
7. Submission passes format validation: done (probability and label formats).
8. Upload and public score in the ledger: done (uploads on improvement, `aac submit`).
9. Inside token, time, and submission budgets: done; the daily quota stopped uploads
   gracefully, the wall clock stops new work, tokens are charged per call.
10. `aac ledger` shows a readable history: done (runs, branches, experiments, notes).
M12 closed 2026-09-06 evening with one fully unattended `aac run` per config on the final
code (quota cap 10, local timeout 180 s, S6E9 at one parallel researcher):

| Run | Wall time | Pool | Blend OOF | Uploads | Public |
|---|---|---|---|---|---|
| spaceship-titanic 20260906-1838 | 17 min | 11 members | 0.81387 accuracy (hill climb of 2) | 2 | 0.80056 |
| playground-series-s6e9 20260906-1859 | 117 min | 13 members | 0.94201 AUC (logistic stack) | 1 | 0.94178 |

Both inside the 180-minute wall clock and the daily quota; every upload and experiment is in
the ledger; `aac ledger` shows the history. On S6E9 the blend beats every single model on OOF
and on the public board (0.94178 vs 0.94162). On Spaceship the day's best public score
(0.8036) came from an earlier single experiment; today's OOF-best blend scored lower publicly,
which is the noisy public slice, and OOF remains the ranking signal. Researcher notes: the
neural track worked on S6E9 under nemotron-super (3/4, best 0.93833) but not on Spaceship
(0/4); the local qwen linear researcher went 2/4 on Spaceship and 0/4 on S6E9. The README's
V1 sections 3, 9, and 11 now carry a note pointing to section 17, and the doctor warns on
memory and thread oversubscription.


### 13. Track-safe knowledge, pitfalls memory, degenerate check (2026-09-08)

Diagnosis from the S6E9 run log (runs 1114-2 to 0804). The local qwen linear Researcher went
6/6 in the two runs before the Historian and Scholar notes existed and 1/12 in the three runs
after: its replies imported LogisticRegression first, then HistGradientBoosting, torch,
TabNet, lightgbm, xgboost and catboost, trying to honour a linear track and a GBDT-only
knowledge base at once. The neural seat's two lightgbm/xgboost rounds in 0804 were the same
failure. Every 0804 module from every seat also carried the same interaction-feature block,
copied from the Historian's "best of them" code. Six of the twelve failed rounds in 0804
came from this; four more were API slips already seen in earlier runs; the neural "success"
predicted a constant (AUC 0.5000) and was counted as ok.

- **Track-safe notes.** `notes.track` column (ledger schema v5, migrated in place). The
  Historian writes one `prior` note per configured track holding only experiments whose
  module passes the track's import rules (label fallback when the module is gone), with that
  track's best code. `render_notes(track=)` strips from every note the fenced modules that
  import a forbidden library and the bullet ideas about one (`strip_off_track`), and reports
  how many it dropped. Tip sharing sends an off-track leader module as hypothesis only.
  Prompt v5 says so explicitly. `open` seats see everything, as before.
- **Pitfalls memory.** `Ledger.failed_experiments`; `historian.pitfalls` renders the distinct
  error tails (exception line, carets and file lines removed) with counts, most recent first,
  timeouts with their hypothesis. One shared note for error/timeout/leak/degenerate, one
  note per track for its own import violations. Written once per run; read every round.
- **Degenerate check.** `degenerate_check` after every ok experiment and interview: constant
  predictions or an OOF gain over the class-rate prediction at or below
  `run.degenerate_margin` (0.002) fail with kind `degenerate` and a diagnostic message. The
  score is kept on the failed row; `best_experiments` now requires status ok.
- Tests: track stripping and hypothesis-only tips, per-track priors, pitfalls rendering and
  recording, v4 ledger migration, degenerate rounds end to end, priors per track through the
  orchestrator. Verified against the live ledger: the Kaggle `GetLeaderboard` RPC works with
  the existing client (top 0.94672, 20th 0.94643 on 2026-09-08; our stack 0.94178, rank 687
  of 1210), not yet wired in.

**Not changed, on purpose:** the Assessor still allocates seats from interviews only (the
local model passes its interview); revisit after a run with the fixes above. `run.max_trees`
never applied to Researcher modules, so it is not the envelope constraint; the 1800 s
timeout with three seeds times five folds on 668k rows is.

### 14. Deterministic levers, hub resilience, round-robin seats, packet reuse (2026-09-08)

Run 20260908-1609, the first with milestone 13, finished at OOF 0.941995 / public 0.94177,
the same plateau as the two runs before it (0.942008 / 0.94178 and 0.942000 / 0.94174). The
stack over the five default families alone was 0.94196 at minute nine; the ten-member stack
at minute 142 was 0.941995: two hours of Researchers bought 0.00004. Track violations fell
from six to one, no degenerate round occurred, but the open seat spent 77 minutes on four
failed rounds (38 of them on twelve timed-out hub attempts in round one, then yesterday's
target-encoding IndexError again), and the local linear seat went 0/4 on code quality.

- **Deterministic plan variants** (`models.variants`, `variant_families`,
  `low_cardinality_max`): `b01-categorical` treats low-cardinality integers as categorical
  levels; `b02-encoded` target-encodes categoricals and those integers inside each fold, the
  integers keeping their numeric column (`_target_encode` no longer drops numeric originals).
  On S6E9 that is Age, Number_of_Cars_Owned, both charging-station counts and
  Environmental_Concern_Level. lightgbm and xgboost only: about 45 s each.
- **Seed bags** (`models.seed_bag`, `seed_bag_top`, `seed_bag_max_seconds`): the two best
  tree families under 120 s are retrained with seeds 43 and 44; every replica is a pool
  member. `run_plan_branch` generalises the default branch; `load_branches` reloads every
  branch on resume.
- **Router**: `fallback_backend` / `fallback_model` per backend (S6E9: hub failures go to
  nemotron-super on the hub, never to qwen, which produced no module in three fallover
  rounds); a circuit breaker (`trip_after` 2, `cooldown_seconds` 600) skips a backend after
  two consecutive failed calls instead of spending four attempts per call. Fallback models
  are verified at startup. Non-retryable errors never trip it.
- **Round-robin seats** (`run.schedule`): `run_researcher` runs up to `until_round` and
  rebuilds its patience and failure counters from history (`history_counters`); the
  orchestrator interleaves seats so every seat plays round n before any plays n+1. Each
  prompt carries the minutes of wall clock left (prompt v6). `experiment_timeout_seconds`
  is 900 on S6E9.
- **Remedies in the environment card**: positional alignment of `y_train` (the IndexError
  seen three runs running), category arithmetic, the HistGradientBoosting and scheduler
  keyword slips, focal-loss gamma, the fit_predict signature and column references. The
  pitfalls note now quotes the offending source line next to the exception.
- **Scholar packet reuse** (`scholar.reuse_runs`, ledger schema v6 `notes.origin_run`): the
  latest packets on the slug are copied into a run when they are under three runs old.
  Half of today's 24 ideas were near-duplicates of yesterday's, at 14 minutes of hub time.
- **S6E9 config**: the local qwen linear seat is retired (1 working round in 20 over five
  runs; interviews cannot see it because the interview module is trivial). README section
  13 item 3 amended: the second backend stays configured and verified as the fallback.
- Tests: breaker and fallback pair, config validation, variant plans and family filter,
  stepping and counters, offending line, note origins, packet reuse across runs, seven
  branches end to end, round-robin versus sequential order.

**Expected effect on S6E9:** the encoded and categorical variants answer, at no LLM cost,
whether target encoding and level handling move the third decimal. If they do not, the
remaining gap to 0.9467 is the original dataset, and the Researcher layer is the wrong place
to look.

### 15. Pending public scores, score backfill, `aac scores`, `--repeat` (2026-09-09)

Run 20260908-2031 (the first with milestone 14) ended with `KaggleError: submission 56105975
still PENDING after 600s` and was recorded as failed, although Kaggle scored the upload a
few minutes later at 0.94186, the best public score so far. The morning run 20260909-0746
completed at OOF 0.94206 / public 0.94185 with 20 pool members, reused packets from run
1609, and had no hub failure.

- `wait_for_score(raise_on_timeout=False)` returns the pending row; `upload_submission` logs
  a warning and returns it instead of raising. The submission row keeps a NULL score.
- `backfill_public_scores` (Submitter) fills NULL scores from one `ListSubmissions` call and
  marks a run that failed only on that poll as completed. Called at the start of every run
  and resume, in `aac submit`, and by the new `aac scores --slug` command. `aac ledger` now
  prints the submissions table.
- `aac run --repeat N` starts N runs sequentially; a failed run does not stop the next.
- Tests: non-raising poll, a run whose score stays pending completes and is backfilled by
  the next run, the repeat loop survives a failed run.

Milestone 14 readings from the two runs: `categorical` variant 0.94101 (worse than the
default 0.94181), `encoded` 0.94167 (close but below), seed bags 0.94174 / 0.94175. The
stacker still gained: OOF 0.94209 and 0.94206 against 0.94200 before, public 0.94186 and
0.94185 against 0.94178. Target encoding and level handling are not the half-point lever on
this table.

### 16. Original dataset as extra training rows (2026-09-09)

The owner confirmed the S6E9 data was generated from
`itzzomkar/ev-adoption-behavior-and-range-anxiety` (CC0, 10,000 rows). Its columns match the
competition's exactly apart from `Buyer_ID`; target labels and rates match (17.5% Yes),
the category vocabularies match, no row duplicates a synthetic one, and three numeric
columns carry about 180 NaNs each. On Playground competitions this is the lever behind the
packed top of the leaderboard, and the deterministic variants of milestone 14 had just shown
that encoding tricks are not.

- **Kaggle client**: `list_dataset_files` and `download_dataset` on
  `datasets.DatasetApiService` (`ListDatasetFiles`, `DownloadDataset`; the download is a
  302 to storage like the competition bundle). `download_all` and the dataset download share
  `_download`.
- **Data cache**: `ensure_dataset` under `runs/_data/datasets/{owner}__{slug}/` (zip or
  single file, parquet-cached); `load_extra_train` aligns a frame with train: rename map,
  unknown columns dropped, dtypes coerced, exact duplicates of synthetic rows removed, fresh
  negative ids.
- **Training**: `prepare_matrix(extra=)` builds the extra matrix with the shared category
  vocabulary; `add_flag` appends the source flag; `cross_validate(extra=)` concatenates the
  rows onto each fold's training part after the mask, so OOF and validation stay on the
  competition's rows. `train_plan` passes them through and records `n_extra`.
- **Harness**: the job carries `extra_train`, `extra_y`, `extra_flag`; the runner sends the
  extras through `build_features` and the matrix together with train, splits them off, adds
  the flag, and appends them to `X_train`/`y_train` in every fold and in the determinism
  check; `meta["n_extra"]` and `meta["extra_flag"]` tell the module. Prompt v7 explains it.
- **Orchestrator**: `_load_extra_train` runs after profiling; the sandbox gets
  `sandbox_extra.parquet`; branches, seed bags, Researchers, the leak check and `aac replay`
  all receive the rows; the summary's data row reports them.
- Config: `competition.extra_train: [{dataset, file, rename, dedupe}]`,
  `competition.extra_flag`. S6E9 config points at the dataset.
- Tests: dataset listing and download, cache and alignment, several-file datasets, shared
  vocabulary and flag in CV, trainer, a module that asserts the rows arrive in every fold,
  config validation, and an end-to-end run with a cached second run.

## Reference write-up: gap analysis (2026-09-06)

The README cites the S6E8 1st-place write-up as inspiration. Its text is saved verbatim in
`docs/reference-s6e8-writeup.md`. Milestones 0 to 6 were designed from the README alone; this
table maps the write-up onto the README and the code, and drives milestones 7 to 11.

| Write-up | README / built so far | Gap | Roadmap change |
|---|---|---|---|
| One autonomous agent does everything: reads the competition, downloads, builds an ensemble, submits | The orchestrator does that loop, but the LLM only plans and writes feature code; training and submitting are deterministic (README section 1, 2) | Deliberate. The README forbids hosted agent frameworks and LLM predictors | None. V2 candidate: LLM-authored model code. |
| 380-model diverse ensemble; every single model goes into the pool; a submission after each batch | Ensembler in section 3 (hill climbing, rank average, stacker); submit only when OOF improves (section 6) | Pool is small: at most 5 families per branch. Seed bagging is a cheap deterministic diversity source the write-up's scale implies | M9: pool every family of every promoted branch and iteration; add seed-bagged replicas of the top plans. Keep submit-on-improvement. |
| Two agents "battle" on the best single model and tips flow from the leader to the trailer | N Architects on different backends, but branches never see each other | Real gap: no within-run sharing | M7: the Critic prompt carries the run leaderboard (plan, features, families, OOF per branch) and the leader's plan, and may borrow. M8: later-iteration Architects get the same. |
| Shared knowledge database | Ledger plus Historian priors across runs (M10) | Within-run sharing (above); cross-run is planned | M10 as planned, plus a per-(backend, model) track record. |
| "Interview" ~150 hub LLMs, assess skills, assign jobs | Doctor pings every configured model; branch triples; the CV scoreboard judges | The assessment should be empirical and persistent | M10: track record per model (valid plans, code that ran, best OOF, tokens, latency) and use it to allocate branches in later runs. |
| ChatGPT Pro research packets yielding novel feature ideas | Not allowed (hosted, manual) | Analog: a long-thinking hub model asked only for feature ideas, fed to Architects as priors | Optional in M8: a `research` prompt on the strongest hub model, inside the token budget. |
| Frontier coding agents (Codex, Claude Code) write the models | Local 14B coder plus hub models over raw HTTP | Large capability gap for single-shot code; mitigated by retries, escalation, and the sandbox's feedback | Documented. V2: a frontier coding backend if the constraint is lifted. |
| A single RealMLP wins | Neural nets out of scope (section 15) | V2 candidate | None. |
| Honest OOF; the community warns that public-LB chasing overfits | OOF is the only ranking signal (section 10) | None | None. |

The write-up gives no mechanics (prompts, CV scheme, iteration counts), so there is nothing
to copy at that level. Its scale is also different: several agents for 1.5 weeks against a
run budget of 3 hours here. The roadmap keeps the README's constraints; moving toward the
write-up's literal method (LLM-written model code, neural families, multi-day budgets) is a
scope decision for the owner.

## Design decisions (settled 2026-09-05)
- `build_features` is restricted to target-free, fold-agnostic transforms. Target encoding is a
  plan option applied by the Trainer inside each fold. This resolves the README section 3 vs
  section 10 conflict and makes the leakage tripwire enforceable by construction.
- Optuna does not run in the iteration loop. Hyperparameters come from the Architect; Optuna
  polishes promoted branches once at the end.
- macOS sandboxing is best effort: AST reject list, runtime socket shim, and the target column
  never passed in. No OS-level network isolation.
- Local backend semaphore is 1 because Ollama serialises requests per model.
- Threads, not asyncio, so the loop stays steppable in a debugger.

## Open risks
- `openai/gpt-oss-20b` was retired from the researcher list on 2026-09-06: on the neural track
  it produced three track violations and an error on S6E9 after a failed interview, and 0/4
  on Spaceship; its seat went to a second `nemotron-3-super` (temperature 0.6). The Assessor
  would have dropped it after one more failed interview.
- Kaggle's gateway behaviour (cookie, signed-URL headers) was observed, not documented; if
  uploads start failing, re-read the official client's source first.
- Hub latency for the diverse Architect models (deepseek-v4-flash, kimi-k3) is minutes per
  call; the wall-clock budget must account for it or the branch list must be trimmed.
- 24 GiB RAM: a 14B local model plus two concurrent training branches is the ceiling. Seen
  live 2026-09-06 on S6E9 with `parallel_branches: 2`: heavy compression and background
  shells killed for memory; S6E9 now runs one researcher at a time, Spaceship keeps two.
- CatBoost at 391 s per plan on S6E9 dominates iteration time; consider fewer iterations or
  dropping it from the default family list for large tables.
