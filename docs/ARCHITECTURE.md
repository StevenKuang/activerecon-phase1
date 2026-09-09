# ActiveBench architecture

The reusable unit is an episode configuration and a method implementing
`ActiveAgent`. Neither the simulator nor the method API depends on the frozen
Phase 1 campaign manifest.

```text
installed scene + episode YAML       method factory + options
              |                              |
       DynamicSceneSim             ActiveAgent / RPC worker
              |                              |
              +--------- EpisodeRunner ------+
                              |
                RGB-D, poses, actions, clock
                              |
                 fixed-rate trajectory replay
                              |
               common gsplat reconstruction
                              |
          held-out appearance + geometry + coverage
                              |
               fresh summary / Spark viewer
```

| Component | Contract and code |
|---|---|
| Scene backend | `common/habitat_env.py` and `sim.py`: mesh/GS rendering, navmesh routing, timed distractors |
| Episode | `episode.py`: sensor, start, motion clock, collision and budgets |
| Policy API | `api.py`: `Observation`, `StreamFrame`, `MethodInfo`, `AgentAction`, `ActiveAgent`, `ViewSelector` |
| Policy loading | `registry.py` and `plugins.py`: built-in names and external Python factories |
| Isolation | `rpc.py`: newline-delimited JSON; RGB/depth transported by persisted file paths |
| Acquisition | `runner.py`: budget enforcement, action execution, observations and evaluator-only masks |
| Campaign | `campaign.py` and `scripts/run_campaign.py`: selection, preflight, receipts, stage execution and failure status |
| Reference preparation | `scripts/prepare_evaluation.py`: clean surface and shared cube catalog, before methods run |
| Evaluation | `eval/`: method-independent training, image metrics, sampled geometry and coverage |
| Presentation | `scripts/summarize_benchmark.py`, `web_export.py`, `web_live.py`, `webdemo/` |

## Process boundary

The orchestrator needs only the core Python package. Each episode runs in
`habitat` (mesh) or `habitat-gs` (Gaussian stages); each external planner runs in
its registered or explicitly configured Python environment. Common training
runs in `bencheval`. A CUDA/toolchain conflict in one method need not change
another method's environment. Failed cells are logged and the campaign exits
nonzero after recording the outcomes.

The benchmark persists RGB/depth before sending the observation to a worker.
RPC reconstructs the same typed object there; method logs go to stderr while
stdout is reserved for protocol messages. This transport currently assumes
processes on the same host with access to the same files.

## Output ownership

`benchmark-run.json` identifies one new experiment; `manifest.json` records its
actual acquisition. `reconstructions/<name>/eval.json` describes common training
and evaluation. `summary/` derives fresh results from these outputs. Large
assets and runs are outside Git by default.

`phase1/` is the retained experiment/evidence package. Its runner selects
Phase 1 cells and exact overrides; its report script rebuilds saved tables.
It calls the same stage helpers as the general runner but does not define the
platform's allowed scenes or methods.

The research checkout additionally contains revisit-policy/RL/UAV experiments.
The delivery exports the shared platform and published adapters. The explicit
export manifest records source and delivered hashes; research-only registrations
and backends are the documented exclusions. See [SOURCE_SYNC.md](SOURCE_SYNC.md).
