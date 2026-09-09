# Reproduce the Phase 1 report

This guide concerns the **Phase 1 results**. To benchmark another
scene or method, start with [RUNNING.md](RUNNING.md); it does not require the
Phase 1 archives. Exact Phase 1 settings and caveats are in
[PHASE1_PROTOCOL.md](PHASE1_PROTOCOL.md).

The report distinguishes shared severe/clean PSNR from provisional GS cube
PSNR. Their [sampling and delta definitions](PROTOCOL.md#scene-quality-where-to-evaluate-and-why-six-views)
explain what each measures. Restore the fixed catalogs for report reproduction;
the generic reference builder's current defaults define a new experiment.

Run commands from the repository root. There are three different operations:

| Operation | Actually executed | Requirements |
|---|---|---|
| Verify/regenerate tables | Derive tables from saved evidence JSON | Python standard library |
| Re-score retained models | Render stored Gaussians against archived targets | `bencheval`, GPU, evaluation archive |
| Rerun acquisition/training | Execute planners, collect frames, train new models | Simulator + selected method + evaluator, scene assets and fixed references |

## 1. Verify the saved tables

```bash
python scripts/phase1/report.py --check
# To write the four generated tables/files again:
python scripts/phase1/report.py
```

Expected: `Verified 4 files from primary evidence and model re-scores`.
This checks consistency; it does not run simulation, training or GPU rendering.
For the supplied report, also check its 115 numerical cells:

```bash
python scripts/phase1/check_report_tables.py \
  --report ../../1/final/ActiveBench-Phase1-Report.md
```

Use your actual report path. Expected: `115 numerical report cells; 0 mismatches`.

## 2. Re-render stored models against the report

Install/reuse `bencheval` following [SETUP.md](SETUP.md). Restore the supplied
69.53 GB evaluation archive (models and fixed clean reference images):

```bash
mkdir -p data/phase1
tar -xf ../artifacts/activebench-phase1-evaluation.tar -C data/phase1
python scripts/phase1/data.py --verify data/phase1 \
  --manifest data/phase1/evaluation-manifest.json
```

One scene/method/condition, then every retained report PSNR:

```bash
conda run --no-capture-output -p "$ACTIVEBENCH_ENVS_DIR/bencheval" \
  python scripts/phase1/rescore.py \
  --scene interior_0007 --method r3con-pano --condition d0 \
  --catalog shared --check-report --out outputs/one-result

conda run --no-capture-output -p "$ACTIVEBENCH_ENVS_DIR/bencheval" \
  python scripts/phase1/rescore.py --check-report --out outputs/report-rescore
```

Expected first result: **22.39 dB**, `REPORT MATCH`.
Expected full result: **60 cells; 84 scores; 84 report matches; 0 failures**.
The default tolerance is 0.0001 dB. Inspect `summary.json` and per-view JSON.
A mismatch/missing input exits nonzero. Repeating the command resumes with
model/camera/target/verifier identity checks.

This recomputes PSNR. SSIM, LPIPS, geometry and planning times in the Phase 1
tables remain recorded measurements. The provisional GS cube scores retain the
limitations described in the report; recomputing them does not resolve those
limitations or justify stronger conclusions.

## 3. Run the Phase 1 method matrix again

Install the required simulator and planner environments, weights and simulation
assets from [SETUP.md](SETUP.md). Keep the archived references from step 2.
The Phase 1 runner supports one method, one scene, one condition, or the
whole declared matrix:

```bash
# Plan one cell.
python scripts/phase1/run.py --scene interior_0007 --method r3con-pano --condition d0

# Verified short acquisition + reconstruction recipe, in a separate output.
python scripts/phase1/run.py --scene interior_0007 --method r3con-pano --condition d0 \
  --smoke-seconds 5 --iterations 20 --out runs_phase1_smoke --execute

# Fresh full-budget runs for all 60 retained cells and their methods.
python scripts/phase1/run.py --out runs_phase1_fresh --execute

# Also attempt all four missing/excluded Phase 1 cells: 64 in total.
python scripts/phase1/run.py --include-missing --out runs_phase1_all64 --execute
```

Omit `--execute` for a plan. `--group mesh|gs`, `--scene`, `--method`,
`--condition d0|dyn` and `--seed` filter the frozen manifest. The six method IDs
are `random`, `r3con-pano`, `magician`, `fisherrf`, `gavis`, `gleam`; Phase 1
GLEAM cells are GS-only. Missing/excluded cells remain declared as such in the
retained evidence even if a new attempt succeeds.

The full new 60/64-cell training matrix has not been rerun for this delivery.
Independent acquisition/training need not recover recorded floating-point
scores exactly. Record new outcomes separately. The full method matrix also
requires substantial disk space beyond the source repository.

## Retained streams and visualization

For fixed-stream retraining or Phase 1 playback, extract
`activebench-phase1-streams.tar.gz` into `data/phase1`. Before verifying the
stream manifest, apply the supplied one-frame repair in `artifacts/input-repair/`
with `python ../artifacts/input-repair/apply_stream_repair.py data/phase1`. It repairs one undecodable archived GAVIS PNG using
verified simulator replay; retained model bytes and scores are unchanged.
Then verify `data/phase1/streams-manifest.json` with `scripts/phase1/data.py`.
For Spark, also extract/verify `activebench-phase1-viewer-assets.tar.gz`.
The optional replay archive contains additional planner decision captures.

```bash
# Train a separately named model from a retained acquisition stream.
python scripts/phase1/run.py --scene interior_0007 --method r3con-pano --condition d0 \
  --stage reconstruct --reconstruction-name gsplat-retrained --out data/phase1/runs --execute

# Browse retained final models and recorded acquisition.
python scripts/phase1/view.py --port 8090
```

The Phase 1 viewer validates retained model identity. For new generic runs,
use `scripts/export_web_demo.py --runs-dir ...` as described in
[RUNNING.md](RUNNING.md). [ACCEPTANCE.md](ACCEPTANCE.md) records which checks were
actually executed; [SYSTEM.md](SYSTEM.md) records the validated environment.
