# Run a benchmark

Run commands from the repository root after [SETUP.md](SETUP.md). The orchestrator
needs the core Python dependencies; activate `bencheval` or another core environment.
`ACTIVEBENCH_ENVS_DIR`, `HABITAT_SIM_ROOT` and applicable method source variables
must point to the installed locations. The commands below use the configurable
prefixes supplied through these variables.

## A short, real end-to-end run

This example collects **6 simulation seconds**, prepares a small independent
reference set, and trains for **20 iterations**. Its scores serve as pipeline
diagnostics for this short configuration.

```bash
# 1. Prepare a clean installed scene.
conda run --no-capture-output -p "$ACTIVEBENCH_ENVS_DIR/habitat" \
  python scripts/prepare_scene.py \
  --scene-path "$HABITAT_SIM_ROOT/data/scene_datasets/habitat-test-scenes/van-gogh-room.glb" \
  --name van_gogh --out-dir outputs/tutorial/configs --seconds 6

# 2. Prepare references before running any policy (small smoke recipe).
conda run --no-capture-output -p "$ACTIVEBENCH_ENVS_DIR/habitat" \
  python scripts/prepare_evaluation.py --configs-dir outputs/tutorial/configs \
  --out-dir outputs/tutorial/assets --points 2 --resolution 96 \
  --surface-spacing 8 --surface-points-per-frame 128

# 3. Acquire, resample, train and score.
python scripts/run_campaign.py --configs-dir outputs/tutorial/configs \
  --assets-dir outputs/tutorial/assets --methods random \
  --reconstruction-resolution 160x120 --retrain-iterations 20 \
  --out-dir outputs/tutorial/runs --execute

# 4. Summarize these newly generated results.
python scripts/summarize_benchmark.py --runs-dir outputs/tutorial/runs
```

Expected artifacts:

```text
outputs/tutorial/
  configs/van_gogh__d0__s0.yaml
  assets/assets.json, surface/, eval/van_gogh__s0/
  runs/van_gogh__d0__s0/random/
    benchmark-run.json       # configuration and code/asset identity
    benchmark-status.json    # success/failure and attempts
    episode.yaml             # fully expanded input
    manifest.json            # actual observations, actions, motion, duration
    frames/, stream/         # planner and common reconstruction RGB-D
    coverage.json
    reconstructions/gsplat/
      eval.json, train.log, gaussians.npz
  runs/summary/results.csv, summary.json, RESULTS.md
```

The campaign prints `Finished: 1 completed, 0 failed`. Omit `--execute` to inspect
a plan. For collection alone, replace the asset/reconstruction options with
`--acquisition-only`. A new scene requires reference preparation before common
reconstruction. This workflow generates its own acquisition and reconstruction
artifacts from the installed scene.

## One scene/method, or the full method roster

Any prepared configs can be selected independently:

```bash
python scripts/run_campaign.py --configs-dir outputs/my-study/configs \
  --assets-dir outputs/my-study/assets --methods r3con-pano \
  --scenes my_scene --conditions dyn --seeds 0 \
  --out-dir outputs/my-study/runs --execute

# All five published method adapters plus Random:
python scripts/run_campaign.py --configs-dir outputs/my-study/configs \
  --assets-dir outputs/my-study/assets \
  --methods random r3con-pano magician fisherrf gavis gleam \
  --out-dir outputs/my-study/runs --execute
```

The matrix is **selected YAMLs × selected methods**, defined by your supplied
configurations. GLEAM's Phase 1 validation covers GS scenes; behavior on other
datasets requires validation. The general method settings
are in [configs/methods.yaml](../configs/methods.yaml), and a custom method YAML
can be selected with `--method-config`. Use `--method-options` for JSON option
overrides keyed by method alias. Protocol-owned inputs remain controlled by
the benchmark configuration.

To do a full experiment, prepare a **new** configuration directory using the
300 s defaults and multiple seeds, prepare references with the default sampling
recipe, and omit the short-run reconstruction overrides. Defaults are 1600 × 1200
training streams and 30,000 iterations. This costs substantially more GPU time
and disk space. Estimate full-matrix runtime using representative runs at
the intended budget.

## Resume and inspect failures

Repeat the same campaign command to resume matching outputs. The runner rejects
changed configs, method options/file source, core pipeline code or reference
assets in an existing cell. Choose a new `--out-dir` when changing the recipe.
Keep receipts intact. Resume requires a matching generic receipt, including
for legacy runs. Keep external source/weights/environment
versions fixed and record them separately as described in [PROTOCOL.md](PROTOCOL.md).

A failed cell remains visible in `benchmark-status.json` and
`campaign-status.json`; the process exits nonzero. Inspect
`<cell>/<method>.launcher.log`, `<method>/agent_worker.log`, and
`reconstructions/gsplat/train.log`. The default makes one acquisition attempt;
`--episode-retries N` enables bounded retry of early startup failures.

The summary retains incomplete cells in denominators, reports per-scene seed
mean/std, and records matched dynamic-minus-static deltas in JSON. It refuses
mixed time/motion/camera/reference/reconstruction regimes. Select a compatible
scene group with `--scenes`, or keep different protocols in different run roots.

For this default cube workflow, `psnr` averages all six faces at each evaluation
point; `dyn_minus_d0_psnr` in `summary.json` is the paired change on that same
catalog. See [camera selection and dynamic-impact interpretation](PROTOCOL.md#scene-quality-where-to-evaluate-and-why-six-views).
The Phase 1 report's severe/clean regional contrasts use a separate catalog.

## View a new reconstruction

Add `--export-spark` to a full campaign, or let the exporter prepare the model:

```bash
python scripts/export_web_demo.py --runs-dir outputs/tutorial/runs \
  --reconstruction gsplat --shared-eval-dir outputs/tutorial/assets/eval --prepare-cache
```

Open http://127.0.0.1:8090 and keep the process running. Select a recording,
apply changes, then use timeline playback/stepping and camera controls.
The viewer displays the stored final reconstruction alongside recorded
acquisition. Playback advances the trajectory and RGB frames while the final
Gaussian model remains fixed.

- **Select:** choose scene, condition (`d0` or `dyn`), method, reconstruction and
  seed, then click **Apply changes**.
- **Replay:** use play/pause, frame stepping or the progress slider. RGB and
  trajectory state follow recorded timestamps; disabling the 3DGS layer reveals
  accumulated RGB-D points.
- **Inspect:** **Go to capture** returns to the recorded camera; FPV, orbit/zoom
  and WASD/QE move the inspection camera. Layer controls expose trajectories,
  camera history and evaluation poses.
- **Compare:** enable **compare (split view)**, select B and apply changes. Keep
  condition fixed to compare methods, or method fixed to compare `d0`/`dyn`.
  Same-scene panes share the inspection camera and replay clock.

[Add another scene](SCENES.md) · [Add a policy](adding-a-method.md) ·
[Reproduce the Phase 1 report](REPRODUCING.md)
