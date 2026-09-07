# Reproducing the Phase 1 evidence

There are four distinct operations. Each has a different input and guarantee.

| Operation | Inputs | Command | Meaning |
|---|---|---|---|
| Rebuild tables | Git evidence and verification JSON | `python scripts/phase1/report.py --check` | Exact regeneration of release tables |
| Re-score models | Evaluation archive; bencheval CUDA environment | `python scripts/phase1/rescore.py --out outputs/rescore` | Independent rendering of saved models on frozen catalogs |
| Retrain recorded acquisition | Evaluation + stream archives | `python scripts/phase1/run.py --stage reconstruct --reconstruction-name gsplat-retrained --out data/phase1/runs --execute` | New training on the retained input; GPU numerical variation is possible |
| New acquisition and reconstruction | Simulation assets, method code/checkpoints and evaluation archive | `python scripts/phase1/run.py --out runs_phase1_new --execute` | A new campaign under the recorded settings |

Re-scoring computes PSNR over every evaluation image, using the same renderer,
camera convention and metric as the common evaluator. It writes new files;
historical results and model arrays remain untouched. `--mode recorded` checks
the original per-cell DC regime, `--mode legacy` uses the release's common
regime, and the default `both` records both where they differ. Cube evaluation
always uses legacy DC. Geometry, SSIM and LPIPS in the evidence remain
explicitly historical; use `scripts/retrain_eval.py --eval-only` in a separate
named reconstruction directory for the full metric suite.

Restore the external data beside the code:

```bash
mkdir -p data/phase1
tar -xf /path/to/activebench-phase1-evaluation.tar -C data/phase1
tar -xf /path/to/activebench-phase1-viewer-assets.tar.gz -C data/phase1
# Needed for uniform-time Spark replay and exact recorded-stream retraining:
tar -xf /path/to/activebench-phase1-streams.tar.gz -C data/phase1
# Optional: the planners' additional decision observations.
tar -xf /path/to/activebench-phase1-replay.tar.gz -C data/phase1
python scripts/phase1/data.py --verify data/phase1 \
  --manifest data/phase1/evaluation-manifest.json
python scripts/phase1/data.py --verify data/phase1 \
  --manifest data/phase1/streams-manifest.json
python scripts/phase1/data.py --verify data/phase1 \
  --manifest data/phase1/viewer-assets-manifest.json
```

The evaluation archive contains selected Gaussian models, primary episode
metadata, frozen clean evaluation images/depth/cameras and surface samples.
The optional replay archive adds recorded decision images/depth/masks. Spark
uses the manifest's uniform stream, supplied by the stream archive at 1600 x
1200. The viewer-assets archive supplies the moving-object meshes. Each archive has a per-file SHA-256
manifest. Archives are deliberately separate from Git and from simulation
datasets/checkpoints obtained from their authors.

For a portable re-score, no original simulator or method repository is needed:

```bash
conda run --no-capture-output -n bencheval python scripts/phase1/rescore.py \
  --mode legacy --data data/phase1 --out outputs/rescore
```

To attempt only one cell:

```bash
python scripts/phase1/run.py \
  --cell gs/interior_0007__d0__s0/r3con-pano \
  --out runs_phase1_new --execute
```

Short smoke runs require a distinct output root. They never enter the result
tables:

```bash
python scripts/phase1/run.py --cell gs/interior_0007__d0__s0/random \
  --smoke-seconds 5 --iterations 20 --out runs_smoke --execute
```

Do not regenerate evaluation cameras and call them the same experiment. Use
the archived transforms and targets. `build_shared_eval_set.py` and
`build_uniform_eval_set.py` support new studies; resampling the catalog changes
the question being measured. Do not substitute 640p models, modified-recipe
retries or shorter GAVIS settings into a full-budget cell.

The historical configuration did not save a complete immutable simulator and
CUDA build at acquisition time. The dependency snapshots describe the
currently verified machine. Full fresh acquisition/retraining is not promised
to recover historical floats bit for bit. The release acceptance record
states which levels were actually executed.
