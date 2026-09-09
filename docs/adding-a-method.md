# Add a method

Implement a Python factory returning an `ActiveAgent`. You can use a separate
package/environment; **no registry edit is required**. The same adapter runs
inline or through the benchmark's existing RPC boundary.

## Minimal working example

[examples/methods/spin_agent.py](../examples/methods/spin_agent.py) is an executable
example. It rotates in place to exercise the interface; it is not an exploration
baseline for quality claims.

```python
from activebench.api import AgentAction, MethodInfo

class Agent:
    def info(self):
        return MethodInfo(name="my-method", needs_depth=False, pose_access="none")

    def reset(self, seed, task=None):
        # Initialize all per-episode state and a seeded RNG here.
        pass

    def act(self, observation):
        # Camera-frame yaw in radians; motion time is charged by the runner.
        return AgentAction.move_rel([0.0, 0.0, 0.0], dyaw=0.3)

def build_agent(options):
    return Agent()
```

After preparing a scene as in [RUNNING.md](RUNNING.md), test the supplied example:

```bash
conda run --no-capture-output -p "$ACTIVEBENCH_ENVS_DIR/habitat" \
  python scripts/run_benchmark.py \
  --config outputs/tutorial/configs/van_gogh__d0__s0.yaml \
  --agent examples/methods/spin_agent.py:build_agent \
  --agent-python "$ACTIVEBENCH_ENVS_DIR/bencheval/bin/python" \
  --out outputs/tutorial/direct-spin
```

For a GS config, use `-p "$ACTIVEBENCH_ENVS_DIR/habitat-gs"`.
`--agent-python` selects an isolated interpreter; `--agent-env myenv` selects a
named environment. Omit both for an
inline external agent. Built-in methods choose their own registered environments.
The worker environment needs this package's core dependencies plus your method's
own dependencies; it does not need Habitat. Install with
`/path/to/myenv/bin/python -m pip install --no-deps -e /path/to/this/repo` after
installing the core dependencies (`numpy`, `PyYAML`, `scipy`, `Pillow`).

For a packaged implementation use `--agent my_package.adapter:build_agent` and
install that package in the worker environment. A single-file factory is loaded
by absolute path; use a package for adapters with relative imports.

## Observation contract

| Field | Representation |
|---|---|
| `rgb` | H × W × 3 `uint8` RGB |
| `depth` | H × W metric depth, or `None`; zero is invalid depth |
| `intrinsics` | `CameraIntrinsics(width, height, fx, fy, cx, cy)` |
| `pose` | `CameraPose`, or `None` when `pose_access="none"` |
| `time`, `step` | Elapsed simulation seconds and decision index |
| `task` | Optional episode text; declare `needs_task_text` if used |
| `stream_frames` | Newly available trajectory samples; lazy `load_rgb()` / `load_depth()` and optional pose |
| `rgb_path`, `depth_path` | Persisted local files used by RPC |

Set `needs_depth=True` to receive depth when the episode enables it.
`pose_access="none"` removes both decision and stream poses. Masks identifying
distractors, clean targets and evaluation scores are not policy inputs.
The factory receives benchmark configuration such as seed, start pose, scene
bounds, sensor width/height/HFOV, time/capture limits and applicable candidate
positions. Document which of these privileged inputs your method uses;
observation-pose masking alone does not establish a fully pose-free setting.
MAGICIAN's existing reference-surface feasibility input is a declared exception.

## Actions and coordinates

- `AgentAction.move_to(CameraPose(...))`: request a world-frame target.
- `AgentAction.move_rel([right, down, forward], dyaw=..., dpitch=...)`: camera-frame
  displacement and yaw/pitch offsets, in metres and radians.
- `AgentAction.trajectory(poses, capture_mode="last")`: traverse waypoints, with
  an agent observation at the last waypoint; `"all"` captures each waypoint.
  The independent 1 Hz reconstruction stream is sampled in either mode.
- `AgentAction.capture()`: request another observation without moving.
- `AgentAction.done()`: end acquisition early; this duration is recorded.

World coordinates use +Y up. `CameraPose` looks along local −Z (OpenGL camera
convention); relative displacement uses OpenCV right/down/forward axes.
Use `activebench.convention.pose_to_c2w_cv` and the shared conversion helpers
when integrating a library that expects OpenCV matrices. `CameraPose` and the
YAML `start_pose: [x,y,z,yaw,pitch]` use **radians**; YAML `hfov` and motion
`yaw_rate_deg`/`pitch_rate_deg` use degrees. Do not silently interchange them.

The runner routes/clips movement according to the episode, enforces the clock
and capture budget, and executes actions with its own geometry. A requested
move can fail to reach its target. Derive policy state from the next observation,
not an assumption that the command was executed exactly. Avoid endless zero-time
`capture()` loops when `capture_cost=0`; the capture budget still limits them.

## Run the method in a campaign

The supplied example also has a ready-to-run configuration:

```bash
python scripts/run_campaign.py --configs-dir outputs/tutorial/configs \
  --method-config examples/methods/spin-method.yaml --methods random example-spin \
  --acquisition-only --out-dir outputs/api-example --execute
```

Create a method configuration. Relative `.py` paths are relative to this YAML:

```yaml
schema_version: 1
methods:
  my-method:
    agent: ../my_package/adapter.py:build_agent
    environment: myenv
    version: git-commit-or-release-id
    options:
      turn_degrees: 30
```

Alternatively set `python: /absolute/path/to/python` instead of `environment`.
Factories must accept a JSON-serializable options dictionary. Benchmark-owned
keys (seed, camera, start, bounds, budgets and reference surface) cannot be
changed through the general campaign's method overrides.

```bash
python scripts/run_campaign.py --configs-dir outputs/tutorial/configs \
  --method-config /path/to/my-methods.yaml --methods my-method \
  --acquisition-only --out-dir outputs/my-method-test --execute
```

For common reconstruction, replace `--acquisition-only` with the same
`--assets-dir` and reconstruction settings used by the other methods. Reuse the
same scene/seed/condition and budget. Save method source revision, dependency
versions and checkpoint hashes with the results. File factories are hashed by
the runner; packaged method contents need your explicit version/provenance.

## Next-best-view selectors and verification

A selector can implement `select(history, candidates) -> int` and return an
index into the supplied `CameraPose` list. Wrap it in `PoolNBVAgent`, returned
by your factory. The existing FisherRF/GAVIS factories demonstrate local
observed-free-space pools; a fixed pool is an explicit alternative. Candidate
sets and information access affect comparability and must be reported.

Start with `python -m pytest tests/test_platform_campaign.py tests/test_bench_rpc.py`.
These tests check actual external-file loading, RPC round trips, pose masking,
invalid contracts and recipe protection. Then run a short real scene, inspect
`agent_worker.log`, `manifest.json` and recorded actions, and only then increase
the budget. API conformance does not validate your method's mathematical
implementation or its upstream paper claims.
