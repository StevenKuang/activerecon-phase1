# Reproduce the report, or run one scene and method

Run these commands from the repository root. The report's stored-model PSNR
results can be reproduced with **only the `bencheval` environment**; simulator
and planner environments are needed only for new acquisition.

## 1. Verify the report tables — no GPU or data required

```bash
python scripts/phase1/report.py --check
```

Expected: `Verified 4 files from primary evidence and model re-scores`.
`phase1/RESULTS.md` reproduces the static means (§4.2), mesh results (§4.4)
and paired regional changes (§5.3–5.4). `results.csv` and `pairs.csv` contain
full precision. Run without `--check` to regenerate these files. For example,
GS static shared PSNR is 23.26 dB for R3-RECON and 19.93 dB for GLEAM;
mesh negative regional contrasts are 17/17. The two protocol groups stay separate.

For the supplied delivery folder layout, check the actual report tables too:

```bash
python scripts/phase1/check_report_tables.py --report ../../1/final/ActiveBench-Phase1-Report.md
```

Expected: `115 numerical report cells; 0 mismatches`. Use your report's path
if it is stored elsewhere.

## 2. Re-render the saved models and check against the report

Prerequisites: Linux x86-64, conda, an NVIDIA GPU/driver compatible with CUDA
12.8, and a C++ compiler. The validated machine has 32 GB VRAM; see
[SYSTEM.md](SYSTEM.md). The first render compiles the gsplat CUDA extension.

If you already have the validated `bencheval` environment, use its prefix.
Otherwise create a **new** prefix; the setup command refuses to overwrite one:

```bash
PHASE1_EVAL_ENV="$HOME/phase1-envs/bencheval"
python scripts/phase1/setup.py --environment bencheval --prefix "$PHASE1_EVAL_ENV" --execute
```

Restore the evaluation archive supplied alongside this repository. This
69.53 GB archive contains all models and clean reference images; acquisition
streams and external method repositories are unnecessary for this step.

```bash
mkdir -p data/phase1
tar -xf ../artifacts/activebench-phase1-evaluation.tar -C data/phase1
python scripts/phase1/data.py --verify data/phase1 --manifest data/phase1/evaluation-manifest.json
```

Start with **one scene, one method, one condition**:

```bash
conda run --no-capture-output -p "$PHASE1_EVAL_ENV" python scripts/phase1/rescore.py \
  --scene interior_0007 --method r3con-pano --condition d0 \
  --catalog shared --check-report --out outputs/one-result
```

Expected: **22.39 dB**, `REPORT MATCH`, then
`1 cells; 1 scores; 1 report matches; 0 failures`.
This checks model/camera identity, every view and the class means, with
`--tolerance-db 0.0001`. A mismatch, unknown selection or missing input returns
a nonzero exit code. Frozen results are never used as the output directory.

Reproduce **every primary and provisional cube PSNR result in the report**:

```bash
conda run --no-capture-output -p "$PHASE1_EVAL_ENV" python scripts/phase1/rescore.py \
  --check-report --out outputs/report-rescore
```

Expected: `60 cells; 84 scores; 84 report matches; 0 failures`.
Inspect `outputs/report-rescore/summary.json` and the per-view JSON files.
Resume by repeating the command; cached results are checked again, and changed
models, cameras, target-image bytes or verifier code invalidate the cache.
`--group mesh` or `--group gs` selects one protocol group.

This command recomputes PSNR. SSIM, LPIPS, geometry and recorded planning times
in the tables remain explicitly historical measurements. Rebuilding a model
or resampling evaluation cameras is a different experiment.

## 3. Acquire and reconstruct one scene with one method

Install only the needed runtimes using [SETUP.md](SETUP.md), plus that scene's
simulation assets and the selected planner's source/checkpoint. Set
`ACTIVEBENCH_ENVS_DIR` if your named environments are outside `~/miniconda3/envs`.

| Operation | Required conda environments |
|---|---|
| Saved-model re-score | `bencheval` only |
| GS scene with R3-RECON, including reconstruction | `habitat-gs`, `r3con`, `bencheval` |
| Mesh scene with R3-RECON, including reconstruction | `habitat`, `r3con`, `bencheval` |
| Random baseline | Scene simulator and `bencheval`; no planner environment |
| Another planner | Scene simulator, that planner's environment, and `bencheval` |

Inspect the selected full-budget run before executing it:

```bash
python scripts/phase1/run.py --scene interior_0007 --method r3con-pano --condition d0
```

Expected: `1 cells; plan only; 0 failures`. The following **verified short
end-to-end example** acquires 5 simulation seconds, resamples six RGB-D frames
to 1600 x 1200, trains for 20 iterations, scores and exports a Spark model:

```bash
python scripts/phase1/run.py --scene interior_0007 --method r3con-pano --condition d0 \
  --smoke-seconds 5 --iterations 20 --out runs_tutorial_smoke --execute
```

Expected: `1 cells; executed; 0 failures`. Model and score files are under
`runs_tutorial_smoke/gs/interior_0007__d0__s0/r3con-pano/reconstructions/gsplat/`.
Smoke scores are excluded from the report. For the full recorded recipe, use
a new output root and omit the two smoke overrides:

```bash
python scripts/phase1/run.py --scene interior_0007 --method r3con-pano --condition d0 \
  --out runs_one_full --execute
```

Use `--scene apartment_1` for the mesh example; `--condition dyn` selects moving
objects. Methods: `r3con-pano`, `magician`, `fisherrf`, `gavis`, `gleam`, `random`.
GLEAM is GS-only. Omit `--condition` to run both static and dynamic cells.
`--stage acquire`, `reconstruct` or `export` runs one pipeline stage;
`--include-missing` also attempts historically missing/excluded cells.
All budgets and per-cell overrides come from `phase1/campaign.json`.

The full 60-cell fresh acquisition/training matrix has not been rerun for this
release. New acquisition and training need not recover historical floats bit
for bit. The [acceptance record](ACCEPTANCE.md) distinguishes the full PSNR
verification, clean evaluator setup, and bounded new-run checks.

## Recorded-stream retraining and Spark

Restore `activebench-phase1-streams.tar.gz` into `data/phase1/` for fixed-stream
retraining or Spark. Spark also needs `activebench-phase1-viewer-assets.tar.gz`.
Verify each with `data.py --verify data/phase1 --manifest data/phase1/streams-manifest.json`
or `viewer-assets-manifest.json`. The optional `replay` archive contains
additional planner decision captures and is unnecessary for uniform-time replay.

```bash
# A new model from the retained stream; preserve the frozen model named gsplat.
python scripts/phase1/run.py --scene interior_0007 --method r3con-pano --condition d0 \
  --stage reconstruct --reconstruction-name gsplat-retrained --out data/phase1/runs --execute
# Browse the retained final models and their recorded acquisition.
python scripts/phase1/view.py --port 8090
```

An existing run refuses a changed configuration or recipe: use a fresh output
root for another experiment. Keep the archived evaluation cameras and targets;
regenerating them changes the measurement. [README](../README.md) describes
Spark playback and comparison controls.
